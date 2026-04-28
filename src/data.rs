use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Zero-copy view into training data.
pub struct DataView<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
}

impl<'a> DataView<'a> {
    pub fn new(x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> Self {
        debug_assert_eq!(
            x.nrows(),
            y.len(),
            "X rows ({}) must match Y length ({})",
            x.nrows(),
            y.len()
        );
        Self { x, y }
    }

    pub fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }
}

/// Owned data for storing in long-lived structs (e.g. Python bindings).
#[derive(Clone, Debug)]
pub struct OwnedData {
    pub x: Array2<f64>,
    pub y: Array1<f64>,
}

impl OwnedData {
    pub fn new(x: Array2<f64>, y: Array1<f64>) -> Self {
        debug_assert_eq!(x.nrows(), y.len());
        Self { x, y }
    }

    pub fn view(&self) -> DataView<'_> {
        DataView::new(self.x.view(), self.y.view())
    }

    pub fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }
}
