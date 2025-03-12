use clap::{Parser, ValueEnum};

use candle_core::{DType, Result, Tensor, D, IndexOp};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
    fn weight(&self) -> Result<&Tensor>;
    fn bias(&self) -> Result<&Tensor>;
}

struct LinearModel {
    linear: Linear,
}

fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;
    let bs = vs.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}

impl Model for LinearModel {
    fn new(vs: VarBuilder) -> Result<Self> {
        let linear = linear_z(IMAGE_DIM, LABELS, vs)?;
        Ok(Self { linear })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }

    fn weight(&self) -> Result<&Tensor> {
        Ok(self.linear.weight())
    }

    fn bias(&self) -> Result<&Tensor> {
        Ok(self.linear.bias().unwrap())
    }
}

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
}

fn model_train<M: Model>(model: M,
                         test_images: &Tensor,
                         test_labels: &Tensor,
                         train_images: &Tensor,
                         train_labels: &Tensor,
                         mut sgd: SGD, epochs: usize) -> Result<M> {
    for epoch in 1..epochs {
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    Ok(model)
}

fn training_loop<M: Model>(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let train_images = m.train_images.to_device(&dev)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // Train 1st model
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model_1 = model_train(M::new(vs.clone())?, &test_images.i(..test_images.shape().dims()[0]/2)?,
                              &test_labels.i(..test_labels.shape().dims()[0]/2)?,
                              &train_images.i(..train_images.shape().dims()[0]/2)?,
                              &train_labels.i(..train_labels.shape().dims()[0]/2)?,
                              SGD::new(varmap.all_vars(), args.learning_rate)?, args.epochs)?;

    // Train 2nd model
    let varmap2 = VarMap::new();
    let vs2 = VarBuilder::from_varmap(&varmap2, DType::F32, &dev);
    let model_2 = model_train(M::new(vs2.clone())?,
                              &test_images.i(test_images.shape().dims()[0]/2..)?,
                              &test_labels.i(test_labels.shape().dims()[0]/2..)?,
                              &train_images.i(train_images.shape().dims()[0]/2..)?,
                              &train_labels.i(train_labels.shape().dims()[0]/2..)?,
                              SGD::new(varmap2.all_vars(), args.learning_rate)?, args.epochs)?;

    // Get the average model
    let model = Linear::new(((model_1.weight()?+ model_2.weight()?)?/2.0)?, Some(((model_1.bias().unwrap()+ model_2.bias().unwrap())?/2.0)?));
    let test_logits = model.forward(&test_images)?;
    let sum_ok = test_logits
        .argmax(D::Minus1)?
        .eq(&test_labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let test_accuracy = sum_ok / test_labels.dims1()? as f32;
    println!("Average test accuracy: {:5.2}%", 100. * test_accuracy);

    Ok(())
}

#[derive(ValueEnum, Clone)]
enum WhichModel {
    Linear,
}

#[derive(Parser)]
struct Args {
    #[clap(value_enum, default_value_t = WhichModel::Linear)]
    model: WhichModel,

    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 100)]
    epochs: usize,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Load the dataset
    let m = candle_datasets::vision::mnist::load()?;

    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    let default_learning_rate = match args.model {
        WhichModel::Linear => 1.,
    };
    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
    };
    match args.model {
        WhichModel::Linear => training_loop::<LinearModel>(m, &training_args),
    }
}