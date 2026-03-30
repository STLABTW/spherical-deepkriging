import logging
import numpy as np
import pandas as pd
import gpboost as gpb
import gc
import warnings

logger = logging.getLogger(__name__)


class UniversalKriging:
    """
    UniversalKriging module using GPBoost for spherical spatial data.
    """
    def __init__(self, num_neighbors, cov_function):
        """
        Initialize Universal Kriging model.
        """
        self.num_neighbors = num_neighbors
        self.gp_model = None
        self.params = None
        self.cov_function = cov_function
        self.has_covariates = False
        self.y_mean = 0.0
        self.nu_was_refitted = False
    
    @staticmethod
    def coords_to_radians(coords):
        """
        Convert (lat, lon) from degrees to radians.
        """
        return np.deg2rad(coords).astype(np.float32)
    
    @staticmethod
    def compute_spherical_distance_matrix(coords_rad):
        """
        Compute exact spherical angular distance matrix.
        """
        lat_rad = coords_rad[:, 0]
        lon_rad = coords_rad[:, 1]
        
        # Compute cosine of angular distance
        cos_theta = (np.sin(lat_rad[:, None]) * np.sin(lat_rad[None, :]) +
                     np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) *
                     np.cos(lon_rad[:, None] - lon_rad[None, :]))
        
        # Compute angular distance (clip to handle numerical errors)
        theta_mat = np.arccos(np.clip(cos_theta, -1.0, 1.0)).astype(np.float32)
        return theta_mat
    
    @staticmethod
    def extract_gp_params(gp_model):
        """
        Extract parameters from GPBoost model.
        """
        cov_pars = gp_model.get_cov_pars()
        
        # Extract nu (don't enforce limit here, just extract)
        if 'Matern_nu' in cov_pars.columns:
            nu = float(cov_pars['Matern_nu'].values[0])
        else:
            nu = 0.5
        
        rho = float(cov_pars['GP_range'].values[0])
        sigma2 = float(cov_pars['GP_var'].values[0])
        nugget = float(cov_pars['Error_term'].values[0])
        return nu, rho, sigma2, nugget
    
    def _get_gpboost_cov_params(self):
        """
        Convert user-friendly covariance function name to GPBoost parameters.
        """
        if self.cov_function == 'exponential':
            return 'matern', 0.5, False
        elif self.cov_function == 'gaussian':
            return 'gaussian', None, False

    def fit(self, coords_train, phi_train, y_train, center_y=True):
        """
        Fit the Universal Kriging model with robust error handling.
        """
        y_train = y_train.reshape(-1).astype(np.float32)
        
        # Center y if requested
        if center_y:
            self.y_mean = np.mean(y_train)
            y_train_centered = y_train - self.y_mean
        else:
            self.y_mean = 0.0
            y_train_centered = y_train
        
        # Convert coordinates to radians
        coords_rad = self.coords_to_radians(coords_train)
        
        # Store some information
        self._coords_rad = coords_rad
        self._phi_train = phi_train.astype(np.float32) if phi_train is not None else None
        self._y_train = y_train_centered
        
        # Determine number of features
        n_features = 0 if phi_train is None else phi_train.shape[1]
        
        # Prepare fit parameters
        fit_params = {
            # "std_dev": True,
            "trace": False
        }
        
        # Get covariance function parameters
        cov_name, cov_shape, estimate_shape = self._get_gpboost_cov_params()
        
        # Case 1: Using a fixed covariance function (exponential, gaussian, etc.)
        if not estimate_shape:
            try:
                # Create GP model with specified covariance function
                if cov_shape is not None:
                    # Matérn with specific shape
                    self.gp_model = gpb.GPModel(
                        gp_coords=coords_rad,
                        cov_function=cov_name,
                        cov_fct_shape=cov_shape,
                        likelihood="gaussian",
                        gp_approx="vecchia",
                        num_neighbors=self.num_neighbors
                    )
                else:
                    # Gaussian or other function without shape parameter
                    self.gp_model = gpb.GPModel(
                        gp_coords=coords_rad,
                        cov_function=cov_name,
                        likelihood="gaussian",
                        gp_approx="vecchia",
                        num_neighbors=self.num_neighbors
                    )
                
                # Fit model
                if phi_train is None:
                    self.has_covariates = False
                    self.gp_model.fit(y=y_train_centered, X=None, params=fit_params)
                else:
                    self.has_covariates = True
                    self.gp_model.fit(y=y_train_centered, X=self._phi_train, params=fit_params)
                
                # Extract parameters
                nu_final, rho_final, sigma2_final, nugget_final = self.extract_gp_params(self.gp_model)
                
                self.nu_was_refitted = False
                nu_est = nu_final
                
            except Exception as e:
                raise RuntimeError(f"Model fitting failed with {self.cov_function} covariance: {str(e)}")
        
        # Case 2: Estimate shape parameter (matern_auto)
        else:
            nu_est = None
            estimation_succeeded = False
            
            try:
                # Create GP model with nu estimation
                gp_model_estimate = gpb.GPModel(
                    gp_coords=coords_rad,
                    cov_function="matern_estimate_shape",
                    likelihood="gaussian",
                    gp_approx="vecchia",
                    num_neighbors=self.num_neighbors
                )
                
                # Fit model
                if phi_train is None:
                    self.has_covariates = False
                    gp_model_estimate.fit(y=y_train_centered, X=None, params=fit_params)
                else:
                    self.has_covariates = True
                    gp_model_estimate.fit(y=y_train_centered, X=self._phi_train, params=fit_params)
                
                # Extract estimated parameters
                nu_est, rho_est, sigma2_est, nugget_est = self.extract_gp_params(gp_model_estimate)
                
                # Check if parameters are valid
                if np.isnan(nu_est) or np.isnan(rho_est) or np.isnan(sigma2_est):
                    raise ValueError("NaN detected in estimated parameters")
                
                estimation_succeeded = True
                
            except Exception as e:
                nu_est = None
                estimation_succeeded = False
            
            # Re-fit if nu > 0.5 or estimation failed
            if not estimation_succeeded or (nu_est is not None and nu_est > 0.5):
                try:
                    # Refit with nu = 0.5
                    self.gp_model = gpb.GPModel(
                        gp_coords=coords_rad,
                        cov_function="matern",
                        cov_fct_shape=0.5,
                        likelihood="gaussian",
                        gp_approx="vecchia",
                        num_neighbors=self.num_neighbors
                    )
                    
                    # Fit model again
                    if phi_train is None:
                        self.gp_model.fit(y=y_train_centered, X=None, params=fit_params)
                    else:
                        self.gp_model.fit(y=y_train_centered, X=self._phi_train, params=fit_params)
                    
                    # Extract new parameters
                    nu_final, rho_final, sigma2_final, nugget_final = self.extract_gp_params(self.gp_model)
                    
                    self.nu_was_refitted = True
                    
                except Exception as e:
                    raise RuntimeError(f"Model fitting failed: {str(e)}")
                
            else:
                # Use the original model
                self.gp_model = gp_model_estimate
                nu_final = nu_est
                rho_final = rho_est
                sigma2_final = sigma2_est
                nugget_final = nugget_est
                
                self.nu_was_refitted = False
            
            # Clean up temporary model if refitted
            if self.nu_was_refitted and estimation_succeeded:
                del gp_model_estimate
                gc.collect()
        
        # Extract beta coefficients
        beta = None
        if n_features > 0:
            try:
                raw_coefs = self.gp_model.get_coef()
                if isinstance(raw_coefs, (pd.Series, pd.DataFrame)):
                    coefs = raw_coefs.to_numpy().flatten()
                else:
                    coefs = np.array(raw_coefs).flatten()
                beta = coefs.reshape(-1, 1)
            except Exception:
                beta = np.zeros((n_features, 1))
        
        # Store parameters
        self.params = {
            'nu': nu_final,
            'rho_rad': rho_final,
            'sigma2': sigma2_final,
            'nugget': nugget_final,
            'beta': beta,
            'nu_initial_estimate': nu_est if nu_est is not None else nu_final,
            'was_refitted': self.nu_was_refitted,
            'cov_function': self.cov_function
        }
        
        # Log fitted parameters
        cov_info = f"Covariance function: {self.cov_function}"
        if self.cov_function == 'gaussian':
            logger.info("%s (Gaussian/RBF)", cov_info)
        elif self.cov_function == 'exponential':
            logger.info("%s (Matérn ν=0.5)", cov_info)
        else:
            logger.info("%s", cov_info)
        
        logger.info(
            "Fitted parameters: nu=%.4f, rho=%.4f, sigma²=%.4f, nugget=%.4f",
            nu_final,
            rho_final,
            sigma2_final,
            nugget_final,
        )
        
        return self
    
    def predict(self, coords_new, phi_new=None, return_centered=True):
        """
        Predict at new locations.
        """
        # Convert coordinates to radians
        coords_rad = self.coords_to_radians(coords_new)
        
        # Make predictions
        if self.has_covariates:
            phi_new = phi_new.astype(np.float32)
            pred = self.gp_model.predict(X_pred=phi_new, gp_coords_pred=coords_rad, predict_var=False)
        else:
            pred = self.gp_model.predict(X_pred=None, gp_coords_pred=coords_rad, predict_var=False)
        
        predictions = pred["mu"]
        
        # Add back mean if requested
        if not return_centered:
            predictions = predictions + self.y_mean
        
        return predictions
    
    def get_coef(self):
        """
        Get fitted coefficients.
        """
        raw_coefs = self.gp_model.get_coef()
        if isinstance(raw_coefs, (pd.Series, pd.DataFrame)):
            coefs = raw_coefs.to_numpy().flatten()
        else:
            coefs = np.array(raw_coefs).flatten()
        
        return coefs
    
    def decompose_prediction(self, coords_new, phi_new):
        """
        Decompose prediction into fixed and random effects.
        """
        # Get total prediction
        y_pred_total = self.predict(coords_new, phi_new, return_centered=True)
        
        # Get coefficients
        coefs = self.get_coef()
        
        # Calculate fixed effect
        n_features = phi_new.shape[1]
        if len(coefs) == n_features:
            y_pred_fixed = phi_new @ coefs
        elif len(coefs) == n_features + 1:
            y_pred_fixed = phi_new @ coefs[:-1] + coefs[-1]
        else:
            raise ValueError(f"Coefficient shape mismatch. Expected {n_features} or {n_features+1}, got {len(coefs)}")
        
        # Calculate random effect
        y_pred_random = y_pred_total - y_pred_fixed
        
        return y_pred_total, y_pred_fixed, y_pred_random
    
    def cleanup(self):
        """Release memory."""
        if self.gp_model is not None:
            del self.gp_model
            self.gp_model = None
        self.params = None
        gc.collect()
