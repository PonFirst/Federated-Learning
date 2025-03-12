use std::collections::HashMap;
use candle_core::{DType, Result, D, IndexOp};
use candle_nn::{loss, ops, Linear, Optimizer, VarBuilder, VarMap, SGD};

use candle_app::{LinearModel, Model};

struct Server {
    clients: HashMap<String, String>,
    models: HashMap<String, LinearModel>,
}

impl Server {
    fn new() -> Self {
        Server {
            clients: HashMap::new(),
            models: HashMap::new(),
        }
    }

    fn register(&mut self, client_ip: String, model: String) {
        println!("Registering client {} for model {}", client_ip, model);
        self.clients.insert(client_ip, model);
    }

    fn init(&mut self, model: String) -> Result<()> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &candle_core::Device::Cpu);
        let global_model = LinearModel::new(vs)?;
        println!("Initializing model {}", model);
        self.models.insert(model, global_model);
        Ok(())
    }

    fn train(&mut self, model_name: String, rounds: usize) -> Result<()> {
        // Load MNIST dataset
        let dataset = candle_datasets::vision::mnist::load()?;
        let dev = candle_core::Device::Cpu;
    
        // Prepare training and test data
        let train_images = dataset.train_images.to_device(&dev)?;
        let train_labels = dataset.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
        let test_images = dataset.test_images.to_device(&dev)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    
        // Initialize or retrieve the global model
        let model = self.models.entry(model_name.clone()).or_insert_with(|| {
            let varmap = VarMap::new();
            let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
            LinearModel::new(vs).unwrap() // Safe to unwrap here for simplicity
        });
    
        // Create optimizer
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut local_model = LinearModel::new(vs)?; // Local copy for training
        let mut sgd = SGD::new(varmap.all_vars(), 1.0)?;
    
        for round in 1..=rounds {
            println!("Round {}", round);
    
            // Train the local model for a few iterations
            for _ in 0..5 {
                let logits = local_model.forward(&train_images)?;
                let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &train_labels)?;
                sgd.backward_step(&loss)?;
            }
    
            // Update the global model with the trained weights
            model.linear = Linear::new(local_model.weight()?.clone(), Some(local_model.bias()?.clone()));
    
            // Evaluate the global model
            let test_logits = model.forward(&test_images)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy = sum_ok / test_labels.dims1()? as f32;
            println!("Round {} test accuracy: {:5.2}%", round, 100. * test_accuracy);
        }
    
        Ok(())
    }

    fn get(&self, model: String) -> Option<(&LinearModel, &str)> {
        self.models.get(&model).map(|m| (m, "ready"))
    }

    fn test(&self, model: String) {
        println!("Testing model {} (not implemented yet)", model);
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut server = Server::new();
    server.init("mnist".to_string())?;
    server.register("127.0.0.1:50001".to_string(), "mnist".to_string());
    server.train("mnist".to_string(), 2)?;
    println!("Server setup complete. Clients: {:?}", server.clients);
    Ok(())
}