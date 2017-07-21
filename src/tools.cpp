#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  
  VectorXd rmse(estimations[0].array().size());
  rmse.fill(0);
  
  // Gather residuals
  for(int i = 0; i < estimations.size(); i++) {
    
    // Compute difference
    VectorXd diff = estimations[i] - ground_truth[i];
    
    // Multiply and add to rmse
    diff = diff.array() * diff.array();
    rmse += diff;
  }
  
  // Get mean
  rmse /= estimations.size();
  
  // Get sqrt
  return rmse.array().sqrt();
  
}
