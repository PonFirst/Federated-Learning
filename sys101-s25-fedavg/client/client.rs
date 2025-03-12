use candle_core::{DType, Result, Tensor, D, IndexOp};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap, SGD};
use candle_datasets::vision::Dataset;
use candle_app::{LinearModel, Model};
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Client {
    server_addr: String,
    model: Option<LinearModel>,
}

impl Client {
    fn new(server_addr: &str) -> Self {
        Client {
            server_addr: server_addr.to_string(),
            model: None,
        }
    }

    async fn join(&mut self, server_ip: &str, _model: &str) -> Result<()> {
        let mut stream = TcpStream::connect(server_ip).await?;
        let message = format!("REGISTER|{}", "127.0.0.1:50001");
        stream.write_all(message.as_bytes()).await?;
        stream.flush().await?;

        let mut buffer = [0; 1024];
        let n = stream.read(&mut buffer).await?;
        let response = String::from_utf8_lossy(&buffer[..n]);
        println!("Server response: {}", response);
        Ok(())
    }

    fn train(&mut self, dataset: &Dataset, epochs: usize, half: bool) -> Result<()> {
        let dev = candle_core::Device::Cpu;
        let train_images = dataset.train_images.to_device(&dev)?;
        let train_labels = dataset.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
        let test_images = dataset.test_images.to_device(&dev)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

        let (train_images, train_labels, test_images, test_labels) = if half {
            (
                train_images.i(..train_images.dims()[0] / 2)?,
                train_labels.i(..train_labels.dims()[0] / 2)?,
                test_images.i(..test_images.dims()[0] / 2)?,
                test_labels.i(..test_labels.dims()[0] / 2)?,
            )
        } else {
            (
                train_images.i(train_images.dims()[0] / 2..)?,
                train_labels.i(train_labels.dims()[0] / 2..)?,
                test_images.i(test_images.dims()[0] / 2..)?,
                test_labels.i(test_labels.dims()[0] / 2..)?,
            )
        };

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut model = LinearModel::new(vs)?;
        let mut sgd = SGD::new(varmap.all_vars(), 1.0)?;

        for epoch in 1..=epochs {
            let logits = model.forward(&train_images)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            sgd.backward_step(&loss)?;

            let test_logits = model.forward(&test_images)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy = sum_ok / test_labels.dims1()? as f32;
            println!(
                "Epoch {}: loss {:8.5}, test acc {:5.2}%",
                epoch,
                loss.to_scalar::<f32>()?,
                100. * test_accuracy
            );
        }
        self.model = Some(model);
        Ok(())
    }

    fn get(&self) -> Option<(&LinearModel, &str)> {
        self.model.as_ref().map(|m| (m, "ready"))
    }

    fn test(&self) {
        println!("Testing local model (not implemented yet)");
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut client = Client::new("127.0.0.1:50051");
    client.join("127.0.0.1:50051", "mnist").await?;
    println!("Client setup complete.");
    Ok(())
}