use std::collections::VecDeque;
use std::rc::Rc;

use numpy::ndarray::{Array, Ix1};

use crate::particle::Particle;

#[derive(Debug)]
pub struct Forest<const MAX_NODES: usize> {
    pub particles: Vec<Particle<MAX_NODES>>,
    pub weights: Vec<f64>,
    pub predictions_buffer: Array<f64, Ix1>,
    pub n_particles: usize,
}

impl<const MAX_NODES: usize> Forest<MAX_NODES> {
    pub fn new(n_particles: usize, n_samples: usize) -> Self {
        Self {
            particles: Vec::with_capacity(n_particles),
            weights: Vec::with_capacity(n_particles),
            predictions_buffer: Array::zeros(n_samples),
            n_particles,
        }
    }

    pub fn reset(&mut self, init_leaf_value: f64, n_samples: usize) {
        self.particles.clear();

        for i in 0..self.n_particles {
            let particle = if i == 0 {
                Particle::new_reference(init_leaf_value, n_samples)
            } else {
                Particle::new(init_leaf_value, n_samples)
            };
            self.particles.push(particle);
        }
    }

    pub fn has_expandable_particles(&self) -> bool {
        self.particles.iter().any(|p| p.has_expandable_nodes())
    }

    /// Resampling now clones both Rc<Tree> and Rc<VecDeque> - but no actual data!
    pub fn resample_particles(&mut self, ancestors: &[usize]) {
        println!("Before resampling:");
        for (i, particle) in self.particles.iter().enumerate() {
            println!(
                "  particle {}: tree count = {}, expandable_nodes: {:?}",
                i,
                Rc::strong_count(&particle.tree),
                particle.expandable_nodes,
            );
        }

        // Now create new particles from the ancestors
        // The old_particles vector is the ONLY owner of the references
        // let mut old_particles = std::mem::take(&mut self.particles);
        // self.particles.clear();

        // let mut new = Vec::with_capacity(ancestors.len());
        // for &idx in ancestors.iter() {
        // new.push(self.particles[idx].clone())
        // }

        // self.particles.clear();
        // self.particles = new;

        self.particles = ancestors
            .iter()
            .map(|&idx| self.particles[idx].clone())
            .collect();

        // old_particles.clear();

        println!("After resampling:");
        for (i, particle) in self.particles.iter().enumerate() {
            println!(
                "  new particle {}: tree count = {}, expandable_nodes: {:?}",
                i,
                Rc::strong_count(&particle.tree),
                particle.expandable_nodes
            );
        }
    }
}
