// op.rs
// To program this i have based the implementation they did on the repo from the
// deep learning systems course by the CMU (repo: https://github.com/dlsyscourse/hw1).
// My implementation is not going to be exactly the same, but i followed a similar approach.
// In rust we do not have the concept of inheritance as in Java or Python so I will handle the operators using a common trait.
use crate::tensor::Tensor;

// All operators in the computational graph implement this trait.
pub trait Operator: std::fmt::Debug {
    // Defines the interface for operators in the computational graph
    // Compute function computes the output in the computational graph.
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String>;

    // Gradient function computes the gradient of the output with respect to the inputs.
    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String>;

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

impl Operator for AddOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("AddOp requires exactly 2 inputs".to_string());
        }
        inputs[0].add(&inputs[1])
    }

    fn gradient(&self, grad_output: &Tensor, _inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // Gradient of addition: both inputs get the same gradient
        Ok(vec![grad_output.clone(), grad_output.clone()])
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

#[derive(Debug, Clone)]
pub struct MulOp;

impl Operator for MulOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("MulOp requires exactly 2 inputs".to_string());
        }
        inputs[0].mul(&inputs[1])
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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

impl Operator for MatMulOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("MatMulOp requires exactly 2 inputs".to_string());
        }
        inputs[0].matmul(&inputs[1])
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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

impl Operator for ReLUOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("ReLUOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].relu())
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // Gradient of ReLU: 1 if input > 0, 0 otherwise
        let mask = Tensor::new_with_device(
            inputs[0].data().mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
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

impl Operator for SumOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("SumOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].sum(self.axis))
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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
pub struct AddScalarOp {
    scalar: f64,
}

impl AddScalarOp {
    pub fn new(scalar: f64) -> Self {
        Self { scalar }
    }
}

impl Operator for AddScalarOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("AddScalarOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].add_scalar(self.scalar))
    }

    fn gradient(&self, grad_output: &Tensor, _inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        Ok(vec![grad_output.clone()])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct MulScalarOp {
    scalar: f64,
}

impl MulScalarOp {
    pub fn new(scalar: f64) -> Self {
        Self { scalar }
    }
}

impl Operator for MulScalarOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("MulScalarOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].mul_scalar(self.scalar))
    }

    fn gradient(&self, grad_output: &Tensor, _inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        Ok(vec![grad_output.mul_scalar(self.scalar)])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct DivOp;

impl Operator for DivOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("DivOp requires exactly 2 inputs".to_string());
        }
        inputs[0].div(&inputs[1])
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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

impl Operator for PowOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("PowOp requires exactly 2 inputs".to_string());
        }
        inputs[0].pow(&inputs[1])
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // d(a^b)/da = b * a^(b-1), d(a^b)/db = a^b * ln(a)
        let a = &inputs[0];
        let b = &inputs[1];

        let b_minus_one = b.add_scalar(-1.0);
        let a_pow_b_minus_one = a.pow(&b_minus_one)?;
        let grad_a = grad_output.mul(&b.mul(&a_pow_b_minus_one)?)?;

        let a_pow_b = a.pow(b)?;
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

impl Operator for ExpOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("ExpOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].exp())
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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

impl Operator for LogOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("LogOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].log())
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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

impl Operator for NegateOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("NegateOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].negate())
    }

    fn gradient(&self, grad_output: &Tensor, _inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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

impl Operator for TransposeOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("TransposeOp requires exactly 1 input".to_string());
        }
        inputs[0].transpose(self.axes.as_deref())
    }

    fn gradient(&self, grad_output: &Tensor, _inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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
    original_shape: Vec<usize>,
}

impl ReshapeOp {
    pub fn new(new_shape: Vec<usize>) -> Self {
        Self {
            new_shape,
            original_shape: Vec::new(), // Will be set during compute
        }
    }
}

impl Operator for ReshapeOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("ReshapeOp requires exactly 1 input".to_string());
        }
        inputs[0].reshape(&self.new_shape)
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // Gradient of reshape is reshape back to original shape
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

impl Operator for BroadcastToOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("BroadcastToOp requires exactly 1 input".to_string());
        }
        inputs[0].broadcast_to(&self.target_shape)
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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

impl Operator for SummationOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("SummationOp requires exactly 1 input".to_string());
        }

        match &self.axes {
            Some(axes) => Ok(inputs[0].sum_axes(Some(axes))),
            None => Ok(inputs[0].sum(None)),
        }
    }

    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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
