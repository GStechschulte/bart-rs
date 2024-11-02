use rand::Rng;
use rand_distr::Uniform;
use std::any::Any;
use std::f64;
use std::iter::Iterator;

// Helper trait for dynamic dispatch
pub trait SplitRule: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn get_split_value_dyn(&self, candidates: &dyn Any) -> Option<Box<dyn Any + Send>>;
    fn divide_dyn(&self, candidates: &dyn Any, split_value: &dyn Any) -> (Vec<usize>, Vec<usize>);
}

// Main trait with associated types
pub trait SplitRuleTyped: Send + Sync {
    type Candidate;
    type SplitValue: Send + 'static;

    fn get_split_value(&self, candidates: &[Self::Candidate]) -> Option<Self::SplitValue>;
    fn divide(
        &self,
        candidates: &[Self::Candidate],
        split_value: &Self::SplitValue,
    ) -> (Vec<usize>, Vec<usize>);
}

// Implement the dynamic trait for any type implementing the typed trait
impl<T: SplitRuleTyped + 'static> SplitRule for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_split_value_dyn(&self, candidates: &dyn Any) -> Option<Box<dyn Any + Send>> {
        let candidates = candidates.downcast_ref::<Vec<T::Candidate>>()?.as_slice();
        self.get_split_value(candidates)
            .map(|value| Box::new(value) as Box<dyn Any + Send>)
    }

    fn divide_dyn(&self, candidates: &dyn Any, split_value: &dyn Any) -> (Vec<usize>, Vec<usize>) {
        if let (Some(candidates), Some(split_value)) = (
            candidates.downcast_ref::<Vec<T::Candidate>>(),
            split_value.downcast_ref::<T::SplitValue>(),
        ) {
            self.divide(candidates.as_slice(), split_value)
        } else {
            (vec![], vec![])
        }
    }
}

pub struct ContinuousSplit;

impl SplitRuleTyped for ContinuousSplit {
    type Candidate = f64;
    type SplitValue = f64;

    fn get_split_value(&self, candidates: &[Self::Candidate]) -> Option<Self::SplitValue> {
        if candidates.len() > 1 {
            let idx = rand::thread_rng().gen_range(0..candidates.len());
            Some(candidates[idx])
        } else {
            None
        }
    }

    fn divide(
        &self,
        candidates: &[Self::Candidate],
        split_value: &Self::SplitValue,
    ) -> (Vec<usize>, Vec<usize>) {
        (0..candidates.len()).partition(|&idx| candidates[idx] <= *split_value)
    }
}

pub struct OneHotSplit;

impl SplitRuleTyped for OneHotSplit {
    type Candidate = i32;
    type SplitValue = i32;

    fn get_split_value(&self, candidates: &[Self::Candidate]) -> Option<Self::SplitValue> {
        if candidates.len() > 1 && !candidates.iter().all(|&x| x == candidates[0]) {
            let idx = rand::thread_rng().gen_range(0..candidates.len());
            Some(candidates[idx])
        } else {
            None
        }
    }

    fn divide(
        &self,
        candidates: &[Self::Candidate],
        split_value: &Self::SplitValue,
    ) -> (Vec<usize>, Vec<usize>) {
        (0..candidates.len()).partition(|&idx| candidates[idx] == *split_value)
    }
}

// /// Interface for split strategies.
// pub trait SplitRule: Send + Sync {
//     fn as_any(&self) -> &dyn Any;
//     fn get_split_value_dyn(&self, candidates: &dyn Any) -> Option<Box<dyn Any + Send>>;
//     fn divide_dyn(&self, candidates: &dyn Any, split_value: &dyn Any) -> (Vec<usize>, Vec<usize>);
// }

// /// Standard continuous split rule. Pick a pivot value and split
// /// depending on if variable is smaller or greater than the value picked.
// pub struct ContinuousSplit;

// impl SplitRule for ContinuousSplit {
//     fn as_any(&self) -> &dyn Any {
//         self
//     }

//     // TODO: Return Result<T, E>
//     fn get_split_value_dyn(&self, candidates: &dyn Any) -> Option<Box<dyn Any + Send>> {
//         // `downcast_ref` is used to recover the concrete type at runtime because of `&dyn Any`
//         if let Some(candidates) = candidates.downcast_ref::<Vec<f64>>() {
//             if candidates.len() > 1 {
//                 let idx = rand::thread_rng().gen_range(0..candidates.len());
//                 Some(Box::new(candidates[idx]))
//             } else {
//                 None
//             }
//         } else {
//             None
//         }
//     }

//     // TODO: Return Result<T, E>
//     fn divide_dyn(&self, candidates: &dyn Any, split_value: &dyn Any) -> (Vec<usize>, Vec<usize>) {
//         if let (Some(candidates), Some(split_value)) = (
//             candidates.downcast_ref::<Vec<f64>>(),
//             split_value.downcast_ref::<f64>(),
//         ) {
//             let (left, right): (Vec<_>, Vec<_>) =
//                 (0..candidates.len()).partition(|&idx| candidates[idx] <= *split_value);
//             (left, right)
//         } else {
//             (vec![], vec![])
//         }
//     }
// }

// /// Choose a single categorical value and branch on it if the variable is that value or not.
// pub struct OneHotSplit;

// impl SplitRule for OneHotSplit {
//     fn as_any(&self) -> &dyn Any {
//         self
//     }

//     // TODO: Return Result<T, E>
//     fn get_split_value_dyn(&self, candidates: &dyn Any) -> Option<Box<dyn Any + Send>> {
//         // `downcast_ref` is used to recover the concrete type at runtime because of `&dyn Any`
//         if let Some(candidates) = candidates.downcast_ref::<Vec<i32>>() {
//             if candidates.len() > 1 && !candidates.iter().all(|&x| x == candidates[0]) {
//                 let idx = rand::thread_rng().gen_range(0..candidates.len());
//                 Some(Box::new(candidates[idx]))
//             } else {
//                 None
//             }
//         } else {
//             None
//         }
//     }

//     // TODO: Return Result<T, E>
//     fn divide_dyn(&self, candidates: &dyn Any, split_value: &dyn Any) -> (Vec<usize>, Vec<usize>) {
//         if let (Some(candidates), Some(&split_value)) = (
//             candidates.downcast_ref::<Vec<i32>>(),
//             split_value.downcast_ref::<i32>(),
//         ) {
//             let (left, right): (Vec<_>, Vec<_>) =
//                 (0..candidates.len()).partition(|&idx| candidates[idx] == split_value);
//             (left, right)
//         } else {
//             (vec![], vec![])
//         }
//     }
// }

/// Choose a random subset of the categorical values and branch on belonging to that set.
///
/// This is the approach taken by Sameer K. Deshpande.
/// flexBART: Flexible Bayesian regression trees with categorical predictors. arXiv,
/// `link <https://arxiv.org/abs/2211.04459>`__
pub struct SubsetSplit;
