// Represents the training data used to fit a Binary Decision Tree
//
// TODO: Use Arrow?

/// Represents a matrix from a vector of data
pub struct Matrix<T>
where
    T: Copy,
{
    data: Vec<T>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl<T> Matrix<T>
where
    T: Copy,
{
    /// Creates a new matrix from a vector of data
    pub fn from_vec(data: Vec<T>, n_rows: usize, n_cols: usize) -> Self {
        if data.len() != (n_rows * n_cols) {
            panic!("Data size does not match the provided dimension");
        }

        Matrix {
            data,
            n_rows,
            n_cols,
        }
    }

    /// Gets the value at the specified row and column
    pub fn get(&self, i: &usize, j: &usize) -> T {
        self.data[i * self.n_cols + j]
    }

    /// Selects the values at the specified rows and column
    pub fn select_rows(&self, rows: &Vec<usize>, col: &usize) -> Vec<T> {
        let mut ret = Vec::<T>::with_capacity(rows.len());

        for i in rows {
            ret.push(self.get(i, col))
        }

        ret
    }
}

pub trait ExternalData {
    fn X(&self) -> &Matrix<f64>;
    fn y(&self) -> &Vec<f64>;
}
