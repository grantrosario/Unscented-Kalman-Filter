#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
  
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.3;
  
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.55;
  
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
  
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
  
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  // class is not initialized yet
  is_initialized_ = false;
  
  ///* State dimension
  n_x_ = 5;
  
  ///* Augmented state dimension
  n_aug_ = 7;
  
  ///* Number of sigma points
  n_sig_  = (2 * n_aug_) + 1;
  
  ///* Sigma point spreading parameter
  lambda_ = 3 - n_x_;
  
  // initial state vector
  x_ = VectorXd(n_x_);
  
  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);
  
  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  
  // NIS values
  NIS_radar_ = 0.;
  NIS_laser_ = 0.;
  
  ///* Laser measurement noise
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_, 0,
              0, std_laspy_;
  
  ///* Weights of sigma points
  weights_ = VectorXd(n_sig_);
  weights_.fill(0.5 / (n_aug_ + lambda_));
  weights_(0) = lambda_/(lambda_+n_aug_);
  
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  if(!is_initialized_) {
    Init(meas_package);
    return;
  }
  
  
  // delta_t
  float delta_t = (meas_package.timestamp_ - prev_timestamp_) / 1000000.0;
  prev_timestamp_ = meas_package.timestamp_;
  
  Prediction(delta_t);
  
  if(use_laser_ and meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  else if(use_radar_ and meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  
  // Augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0);
  x_aug.head(n_x_) = x_;
  
  // Augmented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  
  // Sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  
  // Calculate sqrt of P
  MatrixXd A_ = P_aug.llt().matrixL();
  
  // Create augmented sigma points
  Xsig_aug.col(0) = x_aug; // first column is state vector
  for(int i = 0; i < n_aug_; i++) {
    
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * A_.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A_.col(i);

  }
  
  // Predict sigma points
  for (int i = 0; i< n_sig_; i++)
  {
    //extract values for better readability
    double px       = Xsig_aug(0,i);
    double py       = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    
    // predicted state values
    double px_p, py_p;
    
    // protect against divide by 0
    if(fabs(yawd) > 0.001) {
      px_p = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = py + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = px + v * delta_t * cos(yaw);
      py_p = py + v * delta_t * sin(yaw);
    }
    
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;
    
    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;
    
    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  
  // Compute predicted state mean
  x_.fill(0.0);
  x_ = Xsig_pred_ * weights_;
  
  // Compute predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) +=2.*M_PI;
    
    // covariance
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    
  }
  
  cout << "---- PREDICTION ----" << endl;
  cout << "Time Diff: " << delta_t << endl;
  cout << "x: " << endl << x_ << endl;
  cout << "P: " << endl << P_ << endl << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  
  // number of dimensions
  int n_z_ = 2;
  
  // H matrix
  MatrixXd H_laser_ = MatrixXd(n_z_, 5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;
  
  //measurement covariance matrix - laser
  R_laser_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;
  
  
  // ground truth
  VectorXd z = VectorXd(n_z_);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1);
  
  // compare measurments with predictions
  VectorXd z_pred = H_laser_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  
  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_laser_) * P_;
  
  // NIS
  NIS_laser_ = y.transpose() * S.inverse() * y;

  // DEBUG
  cout << "---- LASER MEASUREMENT ----" << endl;
  cout << "Measurements: " << meas_package.raw_measurements_ << endl;
  cout << "x: " << endl << x_ << endl;
  cout << "P: " << endl << P_ << endl;
  cout << "NIS: " << NIS_laser_ << endl << endl;
  
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // measurement dimension
  int n_z_ = 3;
  
  // Sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, n_sig_);
  
  // ground truth
  VectorXd z = VectorXd(n_z_);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1),
       meas_package.raw_measurements_(2);
  
  // copy over rows
  for(int i = 0; i < n_sig_; i++) {
    
    // extract values for readability
    double px      = Xsig_pred_(0, i);
    double py      = Xsig_pred_(1, i);
    double v       = Xsig_pred_(2, i);
    double yaw     = Xsig_pred_(3, i);
    
    //check for division by 0
    if (fabs(px) < MINIMUM) px = MINIMUM;
    if (fabs(py) < MINIMUM) py = MINIMUM;
    
    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;
    
    // measurement model
    Zsig(0, i) = sqrt(px*px + py*py);                      //rho
    Zsig(1, i) = atan2(py,px);                             //phi
    Zsig(2, i) = (px*v1 + py*v2 ) / sqrt(px*px + py*py);   //rho_dot
  }
  
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);
  
  for(int i = 0; i < n_sig_; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);
  S.fill(0.0);
  
  for(int i = 0; i < n_sig_; i++) {
    
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -=2.*M_PI;
    while (z_diff(1) <-M_PI) z_diff(1) +=2.*M_PI;
    
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  
  // add measurement noise
  MatrixXd R_ = MatrixXd(n_z_, n_z_);
  R_ << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;
  
  // measurement noise covariance matrix
  S += R_;
  
  // cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);
  for(int i = 0; i < n_sig_; i++) {
    
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // Angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -=2.*M_PI;
    while (z_diff(1) <-M_PI) z_diff(1) +=2.*M_PI;
    
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // Angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -=2.*M_PI;
    while (x_diff(3) <-M_PI) x_diff(3) +=2.*M_PI;
    
    // Compute cross correlation
    Tc += (weights_(i) * x_diff * z_diff.transpose());
  }
  
  // Kalman gain K
  MatrixXd K = Tc * S.inverse();
  
  // Residual
  VectorXd z_diff = z - z_pred;
  
  // Angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -=2.*M_PI;
  while (z_diff(1) <-M_PI) z_diff(1) +=2.*M_PI;
  
  // Update state mean and covariance matrix
  x_ += (K * z_diff);
  P_ -= (K * S * K.transpose());
  
  // Compute NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  cout << "---- RADAR MEASUREMENT ----" << endl;
  cout << "Measurements: " << meas_package.raw_measurements_ << endl;
  cout << "x: " << endl << x_ << endl;
  cout << "P: " << endl << P_ << endl;
  cout << "NIS: " << NIS_radar_ << endl << endl;
}

void UKF::Init(MeasurementPackage meas_package) {
  
  if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
    /*
     * Initialize Laser state
     */
    float px = meas_package.raw_measurements_[0];
    float py = meas_package.raw_measurements_[1];
    
    if (fabs(px) < MINIMUM) px = MINIMUM;
    if (fabs(py) < MINIMUM) py = MINIMUM;
    
    x_ << px, py, 0, 0, 0;
  }
  
  else if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    /*
     * Initialize Radar state
     */
    float rho     = meas_package.raw_measurements_[0];
    float phi     = meas_package.raw_measurements_[1];
    float rho_dot = meas_package.raw_measurements_[2];
    
    if (fabs(rho) < 0.001) rho = 0.001;
    
    float px = rho * cos(phi);
    float py = rho * sin(phi);
    float vx = rho_dot * cos(phi);
    float vy = rho_dot * sin(phi);
    
    x_ << px,py,sqrt(vx*vx +vy*vy),0,0;
  }
  
  prev_timestamp_ = meas_package.timestamp_;
  
  // set flag
  is_initialized_ = true;
  
  return;
  
}
