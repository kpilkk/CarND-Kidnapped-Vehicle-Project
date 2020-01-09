/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

// using std::string;
// using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  normal_distribution<double> norm_x(x, std[0]);
  normal_distribution<double> norm_y(y, std[1]);
  normal_distribution<double> norm_theta(theta, std[2]);
  default_random_engine generator;
  num_particles = 100;  // TODO: Set the number of particles
  particles.resize(num_particles);
  for (auto& p : particles) {
    p.x = norm_x(generator);
    p.y = norm_y(generator);
    p.theta = norm_theta(generator);
    p.weight = 1.0;
    weights.push_back(1.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine generator;
  normal_distribution<double> norm_x(0, std_pos[0]);
  normal_distribution<double> norm_y(0, std_pos[1]);
  normal_distribution<double> norm_theta(0, std_pos[2]);

  for (auto & p : particles) {
    if (fabs(yaw_rate) > 0.001) {
      p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      p.theta += yaw_rate * delta_t;
    } else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }


    p.x += norm_x(generator);
    p.y += norm_y(generator);
    p.theta += norm_theta(generator);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (auto & obs : observations) {
    double mindist = std::numeric_limits<float>::max();
    for (auto & pre : predicted) {
      double result = dist(obs.x, obs.y, pre.x, pre.y);
      if (result < mindist) {
        mindist = result;
        obs.id = pre.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (auto & p : particles) {
    std::vector<LandmarkObs> predicted;
    for (const auto& lm : map_landmarks.landmark_list) {
      double result = dist(p.x, p.y, lm.x_f, lm.y_f);
      if (result < sensor_range) {
        LandmarkObs markObs = LandmarkObs { lm.id_i, lm.x_f, lm.y_f };
        predicted.push_back(markObs);
      }
    }

    std::vector<LandmarkObs> observationsmap;
    for (const auto & obs : observations) {
      LandmarkObs tmp;
      tmp.x = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
      tmp.y = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;
      observationsmap.push_back(tmp);
    }

    dataAssociation(predicted, observationsmap);

    p.weight = 1.0;
    for (const auto &obs_m : observationsmap) {
      LandmarkObs landmark;
      for (auto & pre : predicted) {
        if (pre.id == obs_m.id) {
          landmark = pre;
        }
      }

      double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      double exponent = pow(obs_m.x - landmark.x, 2) / (2 * pow(std_landmark[0], 2))
                      + pow(obs_m.y - landmark.y, 2) / (2 * pow(std_landmark[1], 2));
      p.weight *= gauss_norm * exp(-exponent);
    }
    weights.push_back(p.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  int particles_size = particles.size();
  vector<double> weights;
  for (int i = 0; i < particles_size; i++) {
    Particle particle = particles[i];
    weights.push_back(particle.weight);
  }
  
  discrete_distribution<> d(weights.begin(), weights.end());
  default_random_engine generator;
  std::vector<Particle> new_particles;
  for (int i = 0; i < particles_size; i++) {
    int index = d(generator);
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
  weights.clear();
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}