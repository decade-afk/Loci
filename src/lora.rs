

//! # LoRA (Low-Rank Adaptation) Module
//!
//! This module implements LoRA weight merging for dynamic model adaptation.
//! LoRA is a parameter-efficient fine-tuning technique that adds trainable low-rank
//! decomposition matrices to pre-trained model weights, allowing for efficient
//! adaptation without modifying the original model parameters.
//!
//! ## Key Components
//!
//! - **LoRATensor**: Represents tensor data with support for matrix operations
//! - **LoRALayer**: Encapsulates a single LoRA adaptation layer with rank and scaling
//! - **LoRAModel**: Manages a complete LoRA adapter with multiple layers
//! - **LoRAManager**: Handles loading, unloading, and merging multiple LoRA adapters
//!
//! ## Usage
//!
//! LoRA adapters can be loaded from GGUF format files and merged into base model
//! weights. Multiple LoRAs can be stacked with individual scaling factors.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use anyhow::{Result, Context, anyhow};

use crate::gguf::GGUFModel;




/// Represents a tensor used in LoRA computations.
///
/// This struct stores tensor data along with metadata such as name, shape,
/// and data type. It provides methods for common tensor operations including
/// matrix multiplication, scaling, and element-wise addition.
#[derive(Clone)]
pub struct LoRATensor {
    /// Unique identifier for this tensor
    pub name: String,

    /// Tensor dimensions (e.g., [rows, cols] for 2D matrix)
    pub shape: Vec<usize>,

    /// Data type of the tensor elements
    pub dtype: TensorDataType,

    /// Flattened tensor data stored as f32 values
    pub data: Vec<f32>,
}


/// Supported data types for LoRA tensors.
///
/// Defines the numerical precision and format used to store tensor data.
/// Currently supports floating-point and quantized formats.
#[derive(Debug, Clone, Copy)]
pub enum TensorDataType {
    /// 32-bit floating point (full precision)
    F32,
    /// 16-bit floating point (half precision)
    F16,
    /// 8-bit quantized format
    Q8_0,
    /// 4-bit quantized format
    Q4_0,
}

impl LoRATensor {

    /// Returns the total number of elements in the tensor.

    ///

    /// # Returns

    /// The product of all dimension sizes.

    pub fn numel(&self) -> usize {

        self.shape.iter().product()

    }



    /// Performs matrix multiplication with another tensor.

    ///

    /// Computes the matrix product `self @ other` where both tensors must be 2D.

    /// This is a naive implementation using triple nested loops.

    ///

    /// # Arguments

    /// * `other` - The right-hand side tensor for multiplication

    ///

    /// # Returns

    /// A new tensor containing the result of the matrix multiplication

    ///

    /// # Errors

    /// Returns an error if either tensor is not 2D or if dimensions are incompatible

    pub fn matmul(&self, other: &LoRATensor) -> Result<LoRATensor> {

        // Validate that both tensors are 2D matrices

        if self.shape.len() != 2 || other.shape.len() != 2 {

            return Err(anyhow!("matmul requires 2D tensors"));

        }



        // Extract dimensions: self is (m, k), other is (k2, n)

        let m = self.shape[0];

        let k = self.shape[1];

        let k2 = other.shape[0];

        let n = other.shape[1];



        // Validate that inner dimensions match

        if k != k2 {

            return Err(anyhow!("Incompatible shapes for matmul: [{}, {}] @ [{}, {}]", m, k, k2, n));

        }



        // Initialize result matrix with zeros

        let mut result = vec![0.0f32; m * n];



        // Perform matrix multiplication: result[i][j] = sum(self[i][p] * other[p][j])

        for i in 0..m {

            for j in 0..n {

                let mut sum = 0.0;

                for p in 0..k {

                    sum += self.data[i * k + p] * other.data[p * n + j];

                }

                result[i * n + j] = sum;

            }

        }



        Ok(LoRATensor {

            name: format!("{}_x_{}", self.name, other.name),

            shape: vec![m, n],

            dtype: TensorDataType::F32,

            data: result,

        })

    }



    /// Scales all elements in the tensor by a scalar value.

    ///

    /// Multiplies each element in the tensor by the given scalar factor.

    /// This operation is performed in-place.

    ///

    /// # Arguments

    /// * `scalar` - The multiplication factor

    pub fn scale(&mut self, scalar: f32) {

        for val in &mut self.data {

            *val *= scalar;

        }

    }



    /// Adds another tensor to this one element-wise.

    ///

    /// Performs in-place addition: `self = self + other`.

    /// Both tensors must have identical shapes.

    ///

    /// # Arguments

    /// * `other` - The tensor to add

    ///

    /// # Errors

    /// Returns an error if the tensor shapes don't match

    pub fn add_inplace(&mut self, other: &LoRATensor) -> Result<()> {

        if self.shape != other.shape {

            return Err(anyhow!("Incompatible shapes for add: {:?} vs {:?}", self.shape, other.shape));

        }



        for (a, b) in self.data.iter_mut().zip(&other.data) {

            *a += b;

        }



        Ok(())

    }

}




/// Represents a single LoRA adaptation layer.
///
/// A LoRA layer consists of two low-rank matrices (A and B) that are multiplied
/// together to produce a weight update delta. The delta is then scaled by a factor
/// derived from the alpha parameter and rank before being added to the base weights.
///
/// The LoRA decomposition follows: `delta = scale * (B @ A)` where `scale = alpha / rank`
pub struct LoRALayer {
    /// Name of the target layer in the base model
    pub layer_name: String,

    /// Rank of the low-rank decomposition (controls the number of trainable parameters)
    pub rank: usize,

    /// Scaling factor alpha (used to compute the final scaling factor as alpha/rank)
    pub alpha: f32,

    /// The A matrix in the LoRA decomposition (shape: [rank, in_features])
    pub lora_a: LoRATensor,

    /// The B matrix in the LoRA decomposition (shape: [out_features, rank])
    pub lora_b: LoRATensor,
}

impl LoRALayer {

    /// Computes the weight update delta for this LoRA layer.

    ///

    /// Calculates the product `B @ A` and scales it by the provided factor.

    /// The resulting delta can be added to the base model weights to apply the adaptation.

    ///

    /// # Arguments

    /// * `scale` - The scaling factor to apply to the delta

    ///

    /// # Returns

    /// A tensor containing the computed weight update delta

    ///

    /// # Errors

    /// Returns an error if matrix multiplication fails

    pub fn compute_delta(&self, scale: f32) -> Result<LoRATensor> {

        // Compute B @ A to get the low-rank weight update

        let mut delta = self.lora_b.matmul(&self.lora_a)?;

        // Apply the scaling factor

        delta.scale(scale);

        Ok(delta)

    }



    /// Returns the default scaling factor for this LoRA layer.

    ///

    /// The scaling factor is computed as `alpha / rank`, which is the standard

    /// LoRA scaling formula. This helps normalize the contribution of the LoRA

    /// weights regardless of the chosen rank.

    ///

    /// # Returns

    /// The scaling factor as a float

    pub fn get_scaling_factor(&self) -> f32 {

        self.alpha / self.rank as f32

    }

}




/// Represents a complete LoRA adapter model.
///
/// A LoRAModel contains multiple LoRALayer instances that can be applied to
/// different layers of a base model. The adapter is loaded from a GGUF file
/// and can be merged into base model weights.
pub struct LoRAModel {
    /// Unique identifier for this LoRA adapter
    pub id: String,

    /// File system path to the LoRA adapter file
    pub path: PathBuf,

    /// Map of layer names to their corresponding LoRA adaptations
    pub layers: HashMap<String, LoRALayer>,

    /// Default rank used for layers in this adapter
    pub default_rank: usize,

    /// Default alpha value used for layers in this adapter
    pub default_alpha: f32,
}

impl LoRAModel {

    /// Loads a LoRA adapter from a GGUF file.

    ///

    /// Reads the LoRA adapter file and extracts the layer configurations.

    /// Currently implements a simplified loading mechanism.

    ///

    /// # Arguments

    /// * `path` - Path to the GGUF file containing the LoRA adapter

    ///

    /// # Returns

    /// A loaded LoRAModel instance

    ///

    /// # Errors

    /// Returns an error if the file cannot be loaded or parsed

    pub fn load(path: &Path) -> Result<Self> {

        println!("[LoRA] Loading LoRA from {:?}", path);



        // Load the GGUF file containing LoRA weights

        let _gguf = GGUFModel::load(path)

            .context("Failed to load LoRA GGUF")?;



        // Extract ID from filename

        let id = path.file_name()

            .and_then(|n| n.to_str())

            .unwrap_or("unknown")

            .to_string();



        println!("[LoRA] LoRA {} loaded (simplified implementation)", id);



        Ok(Self {

            id,

            path: path.to_path_buf(),

            layers: HashMap::new(),

            default_rank: 8,

            default_alpha: 16.0,

        })

    }



    /// Parses a tensor name to extract layer name and matrix type.

    ///

    /// LoRA tensor names typically follow patterns like "layer_name.lora_A.weight"

    /// or "layer_name.lora_B.weight". This function extracts the layer name and

    /// identifies whether it's an A or B matrix.

    ///

    /// # Arguments

    /// * `name` - The tensor name to parse

    ///

    /// # Returns

    /// A tuple of (layer_name, matrix_type) if the name matches LoRA pattern, None otherwise

    fn parse_tensor_name(name: &str) -> Option<(String, String)> {

        // Look for .lora_A suffix

        if let Some(lora_a_pos) = name.rfind(".lora_A") {

            let layer_name = name[..lora_a_pos].to_string();

            return Some((layer_name, "A".to_string()));

        }



        // Look for .lora_B suffix

        if let Some(lora_b_pos) = name.rfind(".lora_B") {

            let layer_name = name[..lora_b_pos].to_string();

            return Some((layer_name, "B".to_string()));

        }



        None

    }



    /// Merges this LoRA adapter into base model weights.

    ///

    /// Applies the LoRA weight updates to the corresponding layers in the base model.

    /// Each layer's delta is computed and added to the base weights with the specified scale.

    ///

    /// # Arguments

    /// * `base_model_weights` - Mutable reference to the base model's weight map

    /// * `scale` - Scaling factor to apply to the LoRA updates

    ///

    /// # Returns

    /// Ok(()) if merging succeeds

    ///

    /// # Errors

    /// Returns an error if delta computation or weight addition fails

    pub fn merge_into(&self, base_model_weights: &mut HashMap<String, LoRATensor>, scale: f32) -> Result<()> {

        println!("[LoRA] Merging LoRA {} with scale {:.2}", self.id, scale);



        for (layer_name, lora_layer) in &self.layers {

            // Compute the weight update delta for this layer

            let delta = lora_layer.compute_delta(scale)?;



            // Add the delta to the corresponding base model layer

            if let Some(base_weight) = base_model_weights.get_mut(layer_name) {

                // Apply the delta in-place

                base_weight.add_inplace(&delta)?;

                println!("[LoRA]   Merged layer: {}", layer_name);

            } else {

                println!("[LoRA]   Warning: Base layer {} not found", layer_name);

            }

        }



        println!("[LoRA] Merge complete for {} layers", self.layers.len());

        Ok(())

    }



    /// Removes this LoRA adapter's effects from base model weights.

    ///

    /// Computes the negative of the LoRA deltas and applies them to revert the

    /// previous merge operation. This is useful for temporarily disabling a LoRA.

    ///

    /// # Arguments

    /// * `base_model_weights` - Mutable reference to the base model's weight map

    /// * `scale` - Scaling factor that was used during the merge

    ///

    /// # Returns

    /// Ok(()) if unmerging succeeds

    ///

    /// # Errors

    /// Returns an error if delta computation or weight addition fails

    pub fn unmerge_from(&self, base_model_weights: &mut HashMap<String, LoRATensor>, scale: f32) -> Result<()> {

        println!("[LoRA] Unmerging LoRA {}", self.id);



        for (layer_name, lora_layer) in &self.layers {

            // Compute the weight update delta

            let mut delta = lora_layer.compute_delta(scale)?;



            // Negate the delta to reverse the merge

            delta.scale(-1.0);



            // Add the negative delta to the base weights

            if let Some(base_weight) = base_model_weights.get_mut(layer_name) {

                // Apply the negative delta in-place

                base_weight.add_inplace(&delta)?;

                println!("[LoRA]   Unmerged layer: {}", layer_name);

            }

        }



        println!("[LoRA] Unmerge complete");

        Ok(())

    }



    /// Returns statistics about this LoRA adapter.

    ///

    /// Provides information about the number of layers, default parameters,

    /// and total trainable parameters.

    ///

    /// # Returns

    /// A LoRAStats struct containing adapter statistics

    pub fn stats(&self) -> LoRAStats {

        LoRAStats {

            num_layers: self.layers.len(),

            default_rank: self.default_rank,

            default_alpha: self.default_alpha,

            total_params: self.layers.values()

                .map(|l| l.lora_a.numel() + l.lora_b.numel())

                .sum(),

        }

    }

}


/// Statistics about a LoRA adapter.
///
/// Provides metadata about the adapter's configuration and size.
#[derive(Debug)]
pub struct LoRAStats {
    /// Number of layers in the adapter
    pub num_layers: usize,
    /// Default rank used for the adapter
    pub default_rank: usize,
    /// Default alpha value used for the adapter
    pub default_alpha: f32,
    /// Total number of trainable parameters across all layers
    pub total_params: usize,
}




/// Manages multiple LoRA adapters.
///
/// Provides functionality to load, unload, and merge multiple LoRA adapters.
/// Supports stacking multiple adapters with individual scaling factors.
pub struct LoRAManager {
    /// Map of adapter IDs to their corresponding LoRAModel instances
    loras: HashMap<String, LoRAModel>,
}

impl LoRAManager {
    /// Creates a new LoRAManager instance.
    ///
    /// # Returns
    /// A new empty manager ready to load LoRA adapters
    pub fn new() -> Self {
        Self {
            loras: HashMap::new(),
        }
    }

    /// Loads a LoRA adapter from a file and adds it to the manager.
    ///
    /// # Arguments
    /// * `path` - Path to the GGUF file containing the LoRA adapter
    ///
    /// # Returns
    /// The ID of the loaded adapter
    ///
    /// # Errors
    /// Returns an error if the file cannot be loaded
    pub fn load_lora(&mut self, path: &Path) -> Result<String> {
        let lora = LoRAModel::load(path)?;
        let id = lora.id.clone();
        self.loras.insert(id.clone(), lora);
        Ok(id)
    }

    /// Removes a LoRA adapter from the manager.
    ///
    /// # Arguments
    /// * `lora_id` - ID of the adapter to unload
    ///
    /// # Returns
    /// Ok(()) if the adapter was removed
    ///
    /// # Errors
    /// Returns an error if the adapter ID is not found
    pub fn unload_lora(&mut self, lora_id: &str) -> Result<()> {
        if self.loras.remove(lora_id).is_some() {
            println!("[LoRA] LoRA {} unloaded", lora_id);
            Ok(())
        } else {
            Err(anyhow!("LoRA {} not found", lora_id))
        }
    }

    /// Retrieves a reference to a loaded LoRA adapter.
    ///
    /// # Arguments
    /// * `lora_id` - ID of the adapter to retrieve
    ///
    /// # Returns
    /// Some reference to the adapter if found, None otherwise
    pub fn get_lora(&self, lora_id: &str) -> Option<&LoRAModel> {
        self.loras.get(lora_id)
    }

    /// Lists all loaded LoRA adapter IDs.
    ///
    /// # Returns
    /// A vector of adapter IDs currently loaded in the manager
    pub fn list_loras(&self) -> Vec<String> {
        self.loras.keys().cloned().collect()
    }

    /// Merges a stack of LoRA adapters with individual scaling factors.
    ///
    /// Applies multiple LoRA adapters to the base model weights in sequence.
    /// Each adapter can have its own scaling factor, allowing for fine-grained
    /// control over the contribution of each adapter.
    ///
    /// # Arguments
    /// * `lora_ids` - Slice of tuples containing (adapter_id, scale_factor)
    /// * `base_model_weights` - Mutable reference to the base model's weight map
    ///
    /// # Returns
    /// Ok(()) if all adapters are merged successfully
    ///
    /// # Errors
    /// Returns an error if any adapter merge fails
    pub fn merge_stack(&self, lora_ids: &[(String, f32)], base_model_weights: &mut HashMap<String, LoRATensor>) -> Result<()> {
        println!("[LoRA] Merging LoRA stack with {} adapters", lora_ids.len());

        for (lora_id, scale) in lora_ids {
            if let Some(lora) = self.get_lora(lora_id) {
                lora.merge_into(base_model_weights, *scale)?;
            } else {
                println!("[LoRA] Warning: LoRA {} not found, skipping", lora_id);
            }
        }

        Ok(())
    }
}

impl Default for LoRAManager {
    /// Creates a default LoRAManager instance.
    ///
    /// # Returns
    /// A new empty manager
    fn default() -> Self {
        Self::new()
    }
}




/// Creates an example LoRA layer with initialized tensors.
///
/// This function is primarily used for testing and demonstration purposes.
/// It creates a LoRA layer with small constant values in the A and B matrices.
///
/// # Arguments
/// * `layer_name` - Name of the target layer
/// * `rank` - Rank of the low-rank decomposition
/// * `in_features` - Number of input features
/// * `out_features` - Number of output features
///
/// # Returns
/// A LoRALayer with initialized tensors
pub fn create_example_lora_layer(layer_name: &str, rank: usize, in_features: usize, out_features: usize) -> LoRALayer {
    // Create the A matrix (projects from input to rank dimension)
    let lora_a = LoRATensor {
        name: format!("{}.lora_A", layer_name),
        shape: vec![rank, in_features],
        dtype: TensorDataType::F32,
        data: vec![0.01; rank * in_features], // Initialize with small constant values
    };

    // Create the B matrix (projects from rank to output dimension)
    let lora_b = LoRATensor {
        name: format!("{}.lora_B", layer_name),
        shape: vec![out_features, rank],
        dtype: TensorDataType::F32,
        data: vec![0.01; out_features * rank], // Initialize with small constant values
    };

    LoRALayer {
        layer_name: layer_name.to_string(),
        rank,
        alpha: 16.0, // Default alpha value
        lora_a,
        lora_b,
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    /// Tests matrix multiplication functionality.
    #[test]
    fn test_tensor_matmul() {
        let a = LoRATensor {
            name: "A".to_string(),
            shape: vec![2, 3],
            dtype: TensorDataType::F32,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let b = LoRATensor {
            name: "B".to_string(),
            shape: vec![3, 2],
            dtype: TensorDataType::F32,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data.len(), 4);
    }

    /// Tests LoRA layer delta computation.
    #[test]
    fn test_lora_layer_delta() {
        let layer = create_example_lora_layer("test_layer", 4, 512, 512);
        let delta = layer.compute_delta(1.0).unwrap();
        assert_eq!(delta.shape, vec![512, 512]);
    }

    /// Tests LoRA manager functionality.
    #[test]
    fn test_lora_manager() {
        let manager = LoRAManager::new();
        assert_eq!(manager.list_loras().len(), 0);
    }
}
