/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <sstream>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    normal_distribution<double> normalX(0, std[0]);
    normal_distribution<double> normalY(0, std[1]);
    normal_distribution<double> normalTheta(0, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = x + normalX(random_engine);
        p.y = y + normalY(random_engine);
        p.theta = theta + normalTheta(random_engine);
        p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    normal_distribution<double> normalX(0, std_pos[0]);
    normal_distribution<double> normalY(0, std_pos[1]);
    normal_distribution<double> normalTheta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {
        if (fabs(yaw_rate) < 0.0001) {
            particles[i].x += velocity * cos(particles[i].theta) * delta_t;
            particles[i].y += velocity * sin(particles[i].theta) * delta_t;
        } else {
            particles[i].x += velocity / yaw_rate *
                              (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate *
                              (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }

        particles[i].x += normalX(random_engine);
        particles[i].y += normalX(random_engine);
        particles[i].theta += normalTheta(random_engine);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0; i < observations.size(); i++) {
        double minimumDistance = numeric_limits<double>::max();

        int id;
        for (int j = 0; j < predicted.size(); j++) {
            double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

            if (distance < minimumDistance) {
                minimumDistance = distance;
                id = predicted[j].id;
            }
        }

        observations[i].id = id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for (int p = 0; p < num_particles; p++) {
        double partX = particles[p].x;
        double partY = particles[p].y;
        double partTheta = particles[p].theta;

        vector<LandmarkObs> inRangeLandmarks;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            float landmarkX = map_landmarks.landmark_list[j].x_f;
            float landmarkY = map_landmarks.landmark_list[j].y_f;

            bool withinRange = dist(landmarkX, landmarkY, partX, partY) < sensor_range;
            if (withinRange) {
                int landmarkId = map_landmarks.landmark_list[j].id_i;
                inRangeLandmarks.push_back(LandmarkObs{landmarkId, landmarkX, landmarkY});
            }
        }

        vector<LandmarkObs> observationsOnMap;
        for (int j = 0; j < observations.size(); j++) {
            int id = observations[j].id;
            double x = cos(partTheta) * observations[j].x - sin(partTheta) * observations[j].y + partX;
            double y = sin(partTheta) * observations[j].x + cos(partTheta) * observations[j].y + partY;
            observationsOnMap.push_back(LandmarkObs{id, x, y});
        }

        dataAssociation(inRangeLandmarks, observationsOnMap);

        particles[p].weight = 1.0;

        for (int o = 0; o < observationsOnMap.size(); o++) {
            int id = observationsOnMap[o].id;
            double observationX = observationsOnMap[o].x;
            double observationY = observationsOnMap[o].y;

            double landmarkX, landmarkY;
            for (int i = 0; i < inRangeLandmarks.size(); i++) {
                if (inRangeLandmarks[i].id == id) {
                    landmarkX = inRangeLandmarks[i].x;
                    landmarkY = inRangeLandmarks[i].y;
                    break;
                }
            }

            double stdXy = std_landmark[0] * std_landmark[1];
            double stdX2 = std_landmark[0] * std_landmark[0];
            double stdY2 = std_landmark[1] * std_landmark[1];
            double weight = (1. / (2. * M_PI * stdXy)) *
                            exp(-(pow(landmarkX - observationX, 2.) / (2. * stdX2) +
                                  (pow(landmarkY - observationY, 2.) / (2. * stdY2))));

            particles[p].weight *= weight;
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }
    double max_weight = *max_element(weights.begin(), weights.end());

    uniform_real_distribution<double> uniform_real(0.0, 2.0 * max_weight);
    int idx = 0;

    vector<Particle> resampled;
    double beta = 0.0;
    for (int i = 0; i < num_particles; i++) {
        beta += uniform_real(random_engine);
        while (beta > weights[idx]) {
            beta -= weights[idx];
            idx = (idx + 1) % num_particles;
        }
        resampled.push_back(particles[idx]);
    }

    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
