use mnist::Mnist;
use once_cell::sync::Lazy;
use std::io::Read;

pub static MNIST: Lazy<Mnist> = Lazy::new(|| {
    mnist::MnistBuilder::new()
        .base_path("/tmp/mnist/")
        .base_url("http://yann.lecun.com/exdb/mnist/")
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .download_and_extract()
        .finalize()
});

pub fn main() {
    let x = ndarray::ArcArray::from_shape_vec([50_000, 28, 28], MNIST.trn_img.clone())
        .unwrap()
        .map(|v| *v as f32);
    let y = ndarray::ArcArray::from_shape_vec([50_000, 1], MNIST.trn_lbl.clone()).unwrap();

    println!("{:#?}", x.slice(ndarray::s![0, .., ..]));
}
