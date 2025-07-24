use crate::particle::Particle;

#[derive(Clone, Debug)]
pub struct BartState<const MAX_NODES: usize> {
    pub particles: Vec<Particle<MAX_NODES>>,
    pub weights: Vec<f64>,
}

impl<const MAX_NODES: usize> BartState<MAX_NODES> {
    pub fn new(particles: Vec<Particle<MAX_NODES>>, init_weights: Vec<f64>) -> Self {
        Self {
            particles,
            weights: init_weights,
        }
    }
}
