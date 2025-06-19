// op.rs
// To program this i have based the implementation they did on the repo from the
// deep learning systems course by the CMU (repo: https://github.com/dlsyscourse/hw1).
// My implementation is not going to be exactly the same, but i followed a similar approach.
// In rust we do not have the concept of inheritance as in Java or Python so I will handle the operators using a common trait.
use crate::backend::{CPUNumber, GPUFloat, GPUNumber};
use crate::tensor::Tensor;
// All operators in the computational graph implement this trait.
pub trait Operator<T>: std::fmt::Debug
where
    T: GPUNumber,
{
    // Defines the interface for operators in the computational graph
    // Compute function computes the output in the computational graph.
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String>;

    // Gradient function computes the gradient of the output with respect to the inputs.
    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String>;

    // Get number of inputs this operator expects
    fn num_inputs(&self) -> usize;
}

// Basic arithmetic operators
// Curently all operators are only applyable to my implementation of a tensor written over the Ndarray library.
// It is critical to make the tensor interface agnostic to the backend implementation to be able to ue other devices and make this thing more efficient.
// Aditionally, this operations should not be relying on the ndarray library, or either we will have to implement new ones when we
// extend the backend to support other devices like GPU or TPU.
// For now I just did it as best as I could but I know it can be improved from a SWE perspective.
#[derive(Debug, Clone)]
pub struct AddOp;

impl<T> Operator<T> for AddOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("AddOp requires exactly 2 inputs".to_string());
        }
        inputs[0].add(&inputs[1])
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        _inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of addition: both inputs get the same gradient
        Ok(vec![grad_output.clone(), grad_output.clone()])
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

#[derive(Debug, Clone)]
pub struct MulOp;

impl<T> Operator<T> for MulOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("MulOp requires exactly 2 inputs".to_string());
        }
        inputs[0].mul(&inputs[1])
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of multiplication: d(a*b)/da = b, d(a*b)/db = a
        let grad_a = grad_output.mul(&inputs[1])?;
        let grad_b = grad_output.mul(&inputs[0])?;
        Ok(vec![grad_a, grad_b])
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

#[derive(Debug, Clone)]
pub struct MatMulOp;

impl<T> Operator<T> for MatMulOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("MatMulOp requires exactly 2 inputs".to_string());
        }
        inputs[0].matmul(&inputs[1])
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // For C = A @ B:
        // dC/dA = grad_output @ B^T
        // dC/dB = A^T @ grad_output

        let a_data = inputs[0]
            .data()
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let b_data = inputs[1]
            .data()
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let grad_2d = grad_output
            .data()
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        let grad_a = grad_2d.dot(&b_data.t());
        let grad_b = a_data.t().dot(&grad_2d);

        Ok(vec![
            Tensor::new_with_device(grad_a.into_dyn(), inputs[0].device().clone()),
            Tensor::new_with_device(grad_b.into_dyn(), inputs[1].device().clone()),
        ])
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

#[derive(Debug, Clone)]
pub struct ReLUOp;

impl<T> Operator<T> for ReLUOp
where
    T: GPUFloat,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("ReLUOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].relu())
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of ReLU: 1 if input > 0, 0 otherwise
        let zero = <T as CPUNumber>::zero();
        let one = <T as CPUNumber>::one();

        let mask = Tensor::new_with_device(
            inputs[0].data().mapv(|x| if x > zero { one } else { zero }),
            inputs[0].device().clone(),
        );

        let grad = grad_output.mul(&mask)?;
        Ok(vec![grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct SumOp {
    axis: Option<usize>,
}

impl SumOp {
    pub fn new(axis: Option<usize>) -> Self {
        Self { axis }
    }
}

impl<T> Operator<T> for SumOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("SumOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].sum(self.axis))
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of sum: broadcast gradient back to original shape
        let input_shape = inputs[0].shape();

        let broadcasted_grad = match self.axis {
            Some(axis) => {
                // For axis-specific sum, we need to expand dimensions back
                let expanded = grad_output.unsqueeze(axis);
                expanded.broadcast_to(input_shape)?
            }
            None => {
                // Sum over all dimensions: broadcast scalar gradient to input shape
                grad_output.broadcast_to(input_shape)?
            }
        };

        Ok(vec![broadcasted_grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

// Additional operators following the Python course pub structure

#[derive(Debug, Clone)]
pub struct AddScalarOp<T>
where
    T: GPUNumber,
{
    scalar: T,
}

impl<T> AddScalarOp<T>
where
    T: GPUNumber,
{
    pub fn new(scalar: T) -> Self {
        Self { scalar }
    }
}

impl<T> Operator<T> for AddScalarOp<T>
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("AddScalarOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].add_scalar(self.scalar))
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        _inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        Ok(vec![grad_output.clone()])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct MulScalarOp<T>
where
    T: GPUNumber,
{
    scalar: T,
}

impl<T> MulScalarOp<T>
where
    T: GPUNumber,
{
    pub fn new(scalar: T) -> Self {
        Self { scalar }
    }
}

impl<T> Operator<T> for MulScalarOp<T>
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("MulScalarOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].mul_scalar(self.scalar))
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        _inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        Ok(vec![grad_output.mul_scalar(self.scalar)])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct DivScalarOp<T>
where
    T: GPUNumber,
{
    scalar: T,
}

impl<T> DivScalarOp<T>
where
    T: GPUNumber,
{
    pub fn new(scalar: T) -> Self {
        let zero = <T as CPUNumber>::zero();
        if scalar == zero {
            panic!(
                "Cannot create DivScalarOp with zero scalar - this would cause division by zero"
            );
        }
        Self { scalar }
    }
}

impl<T> Operator<T> for DivScalarOp<T>
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("DivScalarOp requires exactly 1 input".to_string());
        }

        let zero = <T as CPUNumber>::zero();
        if self.scalar == zero {
            return Err("Division by zero: scalar divisor is zero".to_string());
        }

        Ok(inputs[0].div_scalar(self.scalar))
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        _inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // For z = a / scalar:
        // dz/da = 1/scalar
        // So grad_a = grad_output * (1/scalar) = grad_output / scalar

        let zero = <T as CPUNumber>::zero();
        if self.scalar == zero {
            return Err("Cannot compute gradient: scalar divisor is zero".to_string());
        }

        Ok(vec![grad_output.div_scalar(self.scalar)])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct DivOp;

impl<T> Operator<T> for DivOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("DivOp requires exactly 2 inputs".to_string());
        }
        inputs[0].div(&inputs[1])
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        let one_over_b = Tensor::ones(inputs[1].shape()).div(&inputs[1])?;
        let grad_a = grad_output.mul(&one_over_b)?;

        let b_squared = inputs[1].mul(&inputs[1])?;
        let neg_a_over_b_squared = inputs[0].negate().div(&b_squared)?;
        let grad_b = grad_output.mul(&neg_a_over_b_squared)?;

        Ok(vec![grad_a, grad_b])
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

#[derive(Debug, Clone)]
pub struct PowOp;

impl<T> Operator<T> for PowOp
where
    T: GPUFloat,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("PowOp requires exactly 2 inputs".to_string());
        }
        inputs[0].powf(&inputs[1])
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // d(a^b)/da = b * a^(b-1), d(a^b)/db = a^b * ln(a)
        let a = &inputs[0];
        let b = &inputs[1];
        let one = <T as CPUNumber>::one();
        let b_minus_one = b.add_scalar(-one);
        let a_pow_b_minus_one = a.powf(&b_minus_one)?;
        let grad_a = grad_output.mul(&b.mul(&a_pow_b_minus_one)?)?;

        let a_pow_b = a.powf(b)?;
        let ln_a = a.log();
        let grad_b = grad_output.mul(&a_pow_b.mul(&ln_a)?)?;

        Ok(vec![grad_a, grad_b])
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

#[derive(Debug, Clone)]
pub struct ExpOp;

impl<T> Operator<T> for ExpOp
where
    T: GPUFloat,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("ExpOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].exp())
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // d(exp(x))/dx = exp(x)
        let exp_x = inputs[0].exp();
        let grad = grad_output.mul(&exp_x)?;
        Ok(vec![grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct LogOp;

impl<T> Operator<T> for LogOp
where
    T: GPUFloat,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("LogOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].log())
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // d(ln(x))/dx = 1/x
        let one_over_x = Tensor::ones(inputs[0].shape()).div(&inputs[0])?;
        let grad = grad_output.mul(&one_over_x)?;
        Ok(vec![grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct NegateOp;

impl<T> Operator<T> for NegateOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("NegateOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].negate())
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        _inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // d(-x)/dx = -1
        Ok(vec![grad_output.negate()])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct TransposeOp {
    axes: Option<Vec<usize>>,
}

impl TransposeOp {
    pub fn new(axes: Option<Vec<usize>>) -> Self {
        Self { axes }
    }
}

impl<T> Operator<T> for TransposeOp
where
    T: GPUNumber
        + Clone
        + std::fmt::Debug
        + rand_distr::num_traits::FromPrimitive
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("TransposeOp requires exactly 1 input".to_string());
        }
        inputs[0].transpose(self.axes.as_deref())
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        _inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of transpose is transpose with inverse permutation
        match &self.axes {
            Some(axes_order) => {
                // Create inverse permutation
                let mut inverse_axes = vec![0; axes_order.len()];
                for (i, &ax) in axes_order.iter().enumerate() {
                    inverse_axes[ax] = i;
                }
                let grad = grad_output.transpose(Some(&inverse_axes))?;
                Ok(vec![grad])
            }
            None => {
                // Default transpose: apply transpose again (since transpose is its own inverse)
                let grad = grad_output.transpose(None)?;
                Ok(vec![grad])
            }
        }
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct ReshapeOp {
    new_shape: Vec<usize>,
    //  original_shape: Vec<usize>, For now we will not store the original shape, but it can be added later if needed.
}

impl ReshapeOp {
    pub fn new(new_shape: Vec<usize>) -> Self {
        Self {
            new_shape,
            //   original_shape: Vec::new(), // Will be set during compute
        }
    }
}

impl<T> Operator<T> for ReshapeOp
where
    T: GPUNumber
        + Clone
        + std::fmt::Debug
        + rand_distr::num_traits::FromPrimitive
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("ReshapeOp requires exactly 1 input".to_string());
        }
        inputs[0].reshape(&self.new_shape)
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of reshape is reshape back to original shape
        // I would like to store the original shape in the operator, but this would require to
        // grab a mutable reference to the Operator, which is not possible in the current design, and also probably not worthy.
        let original_shape = inputs[0].shape();
        let grad = grad_output.reshape(original_shape)?;
        Ok(vec![grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct BroadcastToOp {
    target_shape: Vec<usize>,
}

impl BroadcastToOp {
    pub fn new(target_shape: Vec<usize>) -> Self {
        Self { target_shape }
    }
}

impl<T> Operator<T> for BroadcastToOp
where
    T: GPUNumber
        + Clone
        + rand_distr::num_traits::FromPrimitive
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("BroadcastToOp requires exactly 1 input".to_string());
        }
        inputs[0].broadcast_to(&self.target_shape)
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of broadcast: sum over broadcasted dimensions
        let input_shape = inputs[0].shape();
        let output_shape = grad_output.shape();

        let mut grad = grad_output.clone();

        // Sum over dimensions that were broadcasted
        let ndim_diff = output_shape.len() - input_shape.len();

        // Sum over leading dimensions that were added
        for _ in 0..ndim_diff {
            grad = grad.sum(Some(0));
        }

        // Sum over dimensions that were expanded from size 1
        for (i, (&input_dim, &output_dim)) in input_shape
            .iter()
            .zip(output_shape[ndim_diff..].iter())
            .enumerate()
        {
            if input_dim == 1 && output_dim > 1 {
                grad = grad.sum(Some(i));
                grad = grad.unsqueeze(i);
            }
        }

        Ok(vec![grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct SummationOp {
    axes: Option<Vec<usize>>,
}

impl SummationOp {
    pub fn new(axes: Option<Vec<usize>>) -> Self {
        Self { axes }
    }
}

impl<T> Operator<T> for SummationOp
where
    T: GPUNumber
        + rand_distr::num_traits::FromPrimitive
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("SummationOp requires exactly 1 input".to_string());
        }

        match &self.axes {
            Some(axes) => Ok(inputs[0].sum_axes(Some(axes))),
            None => Ok(inputs[0].sum(None)),
        }
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of sum: broadcast gradient back to original shape
        let input_shape = inputs[0].shape();

        let broadcasted_grad = match &self.axes {
            Some(axes) => {
                let mut grad = grad_output.clone();
                // Expand dimensions back for each summed axis
                for &axis in axes.iter().rev() {
                    grad = grad.unsqueeze(axis);
                }
                grad.broadcast_to(input_shape)?
            }
            None => {
                // Sum over all dimensions: broadcast scalar gradient to input shape
                grad_output.broadcast_to(input_shape)?
            }
        };

        Ok(vec![broadcasted_grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

// Raúl Moldes Castillo: added on Jun-13-2025
// ------------------ ADDITIONAL OPERATORS ------------------ //

/// Minimum operation between two tensors (element-wise).
///
/// Computes the element-wise minimum between two tensors of the same shape.
/// This is essential for implementing clamping operations and numerical stability
/// in loss functions.
///
/// # Mathematical Definition
///
/// ```text
/// min(a, b) = a if a <= b else b
/// ```
#[derive(Debug, Clone)]
pub struct MinOp;

impl<T> Operator<T> for MinOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("MinOp requires exactly 2 inputs".to_string());
        }

        if inputs[0].shape() != inputs[1].shape() {
            return Err(format!(
                "Shape mismatch in MinOp: {:?} vs {:?}",
                inputs[0].shape(),
                inputs[1].shape()
            ));
        }

        inputs[0].min(&inputs[1])
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of min(a, b):
        // ∂min(a,b)/∂a = 1 if a <= b else 0
        // ∂min(a,b)/∂b = 0 if a <= b else 1
        // When a == b, we assign gradient to the first input (arbitrary choice)

        let zero = <T as CPUNumber>::zero();

        let grad_a = Tensor::new_with_device(
            ndarray::Zip::from(inputs[0].data())
                .and(inputs[1].data())
                .and(grad_output.data())
                .map_collect(|&a, &b, &grad| if a <= b { grad } else { zero }),
            inputs[0].device().clone(),
        );

        let grad_b = Tensor::new_with_device(
            ndarray::Zip::from(inputs[0].data())
                .and(inputs[1].data())
                .and(grad_output.data())
                .map_collect(|&a, &b, &grad| if a > b { grad } else { zero }),
            inputs[1].device().clone(),
        );

        Ok(vec![grad_a, grad_b])
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Maximum operation between two tensors (element-wise).
///
/// Computes the element-wise maximum between two tensors of the same shape.
/// This is essential for implementing clamping operations and ReLU-like functions.
///
/// # Mathematical Definition
///
/// ```text
/// max(a, b) = a if a >= b else b
/// ```
#[derive(Debug, Clone)]
pub struct MaxOp;

impl<T> Operator<T> for MaxOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("MaxOp requires exactly 2 inputs".to_string());
        }

        if inputs[0].shape() != inputs[1].shape() {
            return Err(format!(
                "Shape mismatch in MaxOp: {:?} vs {:?}",
                inputs[0].shape(),
                inputs[1].shape()
            ));
        }

        inputs[0].max(&inputs[1])
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of max(a, b):
        // ∂max(a,b)/∂a = 1 if a >= b else 0
        // ∂max(a,b)/∂b = 0 if a >= b else 1
        // When a == b, we assign gradient to the first input (arbitrary choice)

        let zero = <T as CPUNumber>::zero();

        let grad_a = Tensor::new_with_device(
            ndarray::Zip::from(inputs[0].data())
                .and(inputs[1].data())
                .and(grad_output.data())
                .map_collect(|&a, &b, &grad| if a >= b { grad } else { zero }),
            inputs[0].device().clone(),
        );

        let grad_b = Tensor::new_with_device(
            ndarray::Zip::from(inputs[0].data())
                .and(inputs[1].data())
                .and(grad_output.data())
                .map_collect(|&a, &b, &grad| if a < b { grad } else { zero }),
            inputs[1].device().clone(),
        );

        Ok(vec![grad_a, grad_b])
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Clamp operation that constrains values to a specified range.
///
/// Clamps all elements in the input tensor to the range [min_val, max_val].
/// This is crucial for numerical stabilityn loss functions, especially
/// when computing logarithms to avoid log(0) or log(negative).
///
/// # Mathematical Definition
///
/// ```text
/// clamp(x, min_val, max_val) = min(max(x, min_val), max_val)
/// ```
#[derive(Debug, Clone)]
pub struct ClampOp<T>
where
    T: GPUNumber,
{
    min_val: T,
    max_val: T,
}

impl<T> ClampOp<T>
where
    T: GPUNumber,
{
    pub fn new(min_val: T, max_val: T) -> Self {
        if min_val > max_val {
            panic!(
                "ClampOp: min_val ({:?}) cannot be greater than max_val ({:?})",
                min_val, max_val
            );
        }
        Self { min_val, max_val }
    }
}

impl<T> Operator<T> for ClampOp<T>
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("ClampOp requires exactly 1 input".to_string());
        }

        
        Ok(inputs[0].clamp(self.min_val, self.max_val))
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of clamp(x, min_val, max_val):
        // ∂clamp(x)/∂x = 1 if min_val <= x <= max_val else 0

        let zero = <T as CPUNumber>::zero();

        let grad = Tensor::new_with_device(
            ndarray::Zip::from(inputs[0].data())
                .and(grad_output.data())
                .map_collect(|&x, &grad| {
                    if x >= self.min_val && x <= self.max_val {
                        grad
                    } else {
                        zero
                    }
                }),
            inputs[0].device().clone(),
        );

        Ok(vec![grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Square root operation.
///
/// Computes the element-wise square root of the input tensor.
/// This is essential for many mathematical operations including
/// computing norms, standard deviations, and certain activation functions.
///
/// # Mathematical Definition
///
/// ```text
/// sqrt(x) = √x
/// ```
///
/// # Important Notes
///
/// - Input values must be non-negative for real square roots
/// - The gradient is undefined at x = 0 (returns 0 by convention)
/// - For numerical stabilityvery small positive values near 0 should be handled carefully
#[derive(Debug, Clone)]
pub struct SqrtOp;

impl<T> Operator<T> for SqrtOp
where
    T: GPUFloat,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("SqrtOp requires exactly 1 input".to_string());
        }
        // We do not check for negative values here, as this isdone inside the Tensor implementation
        // and it will panic if negative values are present.
        inputs[0].sqrt()
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of sqrt(x): ∂sqrt(x)/∂x = 1/(2*sqrt(x))
        // Special case: at x = 0, we return 0 instead of infinity

        let zero = <T as CPUNumber>::zero();
        let two = <T as CPUNumber>::from_f64(2.0).unwrap();

        let grad = Tensor::new_with_device(
            ndarray::Zip::from(inputs[0].data())
                .and(grad_output.data())
                .map_collect(|&x, &grad_out| {
                    if x == zero {
                        zero // Avoid division by zero
                    } else {
                        let sqrt_x = x.sqrt();
                        grad_out / (two * sqrt_x)
                    }
                }),
            inputs[0].device().clone(),
        );

        Ok(vec![grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Absolute value operation.
///
/// Computes the element-wise absolute value of the input tensor.
/// This is useful for implementing various loss functions and
/// mathematical operations that require non-negative values.
///
/// # Mathematical Definition
///
/// ```text
/// abs(x) = |x| = x if x >= 0 else -x
/// ```
///
/// # Gradient Notes
///
/// The gradient is undefined at x = 0. By convention, we return 0 at this point.
#[derive(Debug, Clone)]
pub struct AbsOp;

impl<T> Operator<T> for AbsOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("AbsOp requires exactly 1 input".to_string());
        }

        Ok(inputs[0].abs())
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        // Gradient of abs(x):
        // ∂abs(x)/∂x = 1 if x > 0, -1 if x < 0, 0 if x = 0 (by convention)

        let zero = <T as CPUNumber>::zero();
        let one = <T as CPUNumber>::one();
        let neg_one = -one;

        let grad = Tensor::new_with_device(
            ndarray::Zip::from(inputs[0].data())
                .and(grad_output.data())
                .map_collect(|&x, &grad_out| {
                    if x > zero {
                        grad_out
                    } else if x < zero {
                        grad_out * neg_one
                    } else {
                        zero // x == 0, gradient undefined, use 0 by convention
                    }
                }),
            inputs[0].device().clone(),
        );

        Ok(vec![grad])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

// Maximum reduction operation along a specified dimension.
///
/// Computes the maximum values along a specified dimension, reducing that dimension.
/// This is essential for CPUNumberally stable softmax computation.
///
/// # Mathematical Definition
///
/// For a tensor x and dimension d:
/// ```text
/// max_along_dim(x, d) = max(x along dimension d)
/// ```
///
/// # Gradient Computation
///
/// The gradient flows only to the elements that achieved the maximum value.
/// For elements that were not the maximum, the gradient is zero.
/// When there are ties (multiple elements have the same maximum value),
/// the gradient is distributed equally among them.
/// 
#[derive(Debug, Clone)]
pub struct MaxAlongDimOp {
    /// Dimension along which to compute the maximum
    dim: usize,
}

impl MaxAlongDimOp {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<T> Operator<T> for MaxAlongDimOp
where
    T: GPUNumber,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("MaxAlongDimOp requires exactly 1 input".to_string());
        }

     
        inputs[0].max_along_dim(self.dim)
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("MaxAlongDimOp requires exactly 1 input".to_string());
        }

        let input = &inputs[0];
        let input_shape = input.shape();

        // First, compute the maximum values again to determine which elements were maximal
        let max_values = input.data().fold_axis(
            ndarray::Axis(self.dim),
            <T as CPUNumber>::min_value(),
            |&acc, &x| if acc > x { acc } else { x },
        );

        // Expand max_values to match input shape for comparison
        let expanded_max = max_values.insert_axis(ndarray::Axis(self.dim));
        let expanded_max_broadcasted = expanded_max
            .broadcast(input_shape)
            .ok_or("Failed to broadcast max values")?;

        // Create mask where input equals max (these get gradients)
        let zero = <T as CPUNumber>::zero();
        let one = <T as CPUNumber>::one();

        let mask = ndarray::Zip::from(input.data())
            .and(&expanded_max_broadcasted)
            .map_collect(|&inp, &max_val| if inp == max_val { one } else { zero });

        // Count how many elements achieved the maximum along each slice
        let count_maxima = mask.sum_axis(ndarray::Axis(self.dim));

        // Expand count to match input shape
        let expanded_count = count_maxima.insert_axis(ndarray::Axis(self.dim));
        let expanded_count_broadcasted = expanded_count
            .broadcast(input_shape)
            .ok_or("Failed to broadcast count")?;

        // Expand grad_output to match input shape
        let expanded_grad = grad_output
            .data()
            .clone()
            .insert_axis(ndarray::Axis(self.dim));
        let expanded_grad_broadcasted = expanded_grad
            .broadcast(input_shape)
            .ok_or("Failed to broadcast gradient")?;

        // Gradient is (mask / count) * grad_output
        // This ensures gradient is distributed equally among tied maxima
        let input_grad = ndarray::Zip::from(&mask)
            .and(&expanded_count_broadcasted)
            .and(&expanded_grad_broadcasted)
            .map_collect(|&mask_val, &count, &grad| {
                if count > zero {
                    mask_val * grad / count
                } else {
                    zero
                }
            });

        Ok(vec![Tensor::new_with_device(
            input_grad,
            input.device().clone(),
        )])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}
/// Softmax operation along a specified dimension.
///
/// Computes the softmax function along the specified dimension with numerical stability
/// This operation combines finding the maximum, subtracting it, exponentiating,
/// summing, and dividing in a single operation for efficiency and numerical stability
///
/// # Mathematical Definition
///
/// ```text
/// softmax(x_i) = exp(x_i - max(x)) / Σⱼ exp(x_j - max(x))
/// ```
///
/// # Gradient Computation
///
/// For softmax, the gradient computation is:
/// ```text
/// ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
/// ```
/// Where δ_ij is the Kronecker delta (1 if i==j, 0 otherwise).
#[derive(Debug, Clone)]
pub struct SoftmaxOp {
    /// Dimension along which to apply softmax
    dim: usize,
}

impl SoftmaxOp {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<T> Operator<T> for SoftmaxOp
where
    T: GPUFloat,
{
    fn compute(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("SoftmaxOp requires exactly 1 input".to_string());
        }

        let input = &inputs[0];
        let input_shape = input.shape();

        // Validate dimension
        if self.dim >= input_shape.len() {
            return Err(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            ));
        }

        let axis = ndarray::Axis(self.dim);

        // Step 1: Find maximum along the specified dimension for numerical stability
        let max_vals = input
            .data()
            .fold_axis(axis, <T as CPUNumber>::min_value(), |&acc, &x| {
                if acc > x { acc } else { x }
            });

        // Step 2: Expand max_vals back to original shape for broadcasting
        let expanded_max = max_vals.insert_axis(axis);
        let broadcasted_max = expanded_max
            .broadcast(input_shape)
            .ok_or("Failed to broadcast max values")?;

        // Step 3: Subtract max and compute exponentials: exp(x - max)
        let shifted_and_exp = ndarray::Zip::from(input.data())
            .and(&broadcasted_max)
            .map_collect(|&x, &max_val| (x - max_val).exp());

        // Step 4: Sum exponentials along the dimension
        let sum_exp = shifted_and_exp.sum_axis(axis);

        // Step 5: Expand sum back to original shape for broadcasting
        let expanded_sum = sum_exp.insert_axis(axis);
        let broadcasted_sum = expanded_sum
            .broadcast(input_shape)
            .ok_or("Failed to broadcast sum values")?;

        // Step 6: Divide by sum to get probabilities
        let result = ndarray::Zip::from(&shifted_and_exp)
            .and(&broadcasted_sum)
            .map_collect(|&exp_val, &sum_val| exp_val / sum_val);

        Ok(Tensor::new_with_device(result, input.device().clone()))
    }

    fn gradient(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("SoftmaxOp requires exactly 1 input".to_string());
        }

        let input = &inputs[0];
        let input_shape = input.shape();

        // Validate dimension
        if self.dim >= input_shape.len() {
            return Err(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            ));
        }

        // Step 1: Find maximum along the specified dimension for numerical stability
        let max_vals = input.max_along_dim(self.dim)?;

        // Step 2: Expand max_vals back to original shape for broadcasting  
        let expanded_max = max_vals.unsqueeze(self.dim);
        let broadcasted_max = expanded_max.broadcast_to(input_shape)?;

        // Step 3: Subtract max and compute exponentials: exp(x - max)
        // This should be fused into a sub operation for efficiency. However curently it is not implemented.
        let negated = broadcasted_max.negate();
        let shifted_input = input.add(&negated)?;

        let exp_vals = shifted_input.exp();

        // Step 4: Sum exponentials along the dimension
        let sum_exp = exp_vals.sum(Some(self.dim));

        // Step 5: Expand sum back to original shape for broadcasting
        let expanded_sum = sum_exp.unsqueeze(self.dim);
        let broadcasted_sum = expanded_sum.broadcast_to(input_shape)?;

        // Step 6: Divide by sum to get probabilities
        let result = exp_vals.div(&broadcasted_sum)?;

        Ok(result)
    }

    fn num_inputs(&self) -> usize {
        1
    }
}
