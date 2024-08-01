import numpy as np

class Metrics:
    def __init__(self, true_rul, mean_rul, upper_bound, lower_bound, alpha):
        self.true_rul = true_rul
        self.mean_rul = mean_rul
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.alpha = alpha


    def compute_cic(self):
        # Initialize CIC to 0
        cic = 0
        # Iterate over all time points
        for self.k in range(len(self.true_rul)):
            # Check if the true RUL is within the estimated CI
            if self.lower_bound[self.k] <= self.true_rul[self.k] <= self.upper_bound[self.k]:
                # If it is, add the ratio of the true RUL and the width of the CI to CIC
                cic += 1 #self.true_rul[self.k] #/ (self.upper_bound[self.k] - self.lower_bound[self.k])
        cic = cic / self.true_rul[0]
        return cic
    
    def compute_cic_25(self):
        # Initialize CIC to 0
        cic_25 = 0
        # Iterate over the last 25% of the time points
        for k in range(int(len(self.true_rul) * 0.75), len(self.true_rul)):
            # Check if the true RUL is within the estimated CI
            if self.lower_bound[k] <= self.true_rul[k] <= self.upper_bound[k]:
                # If it is, add the ratio of the true RUL and the width of the CI to CIC
                cic_25 += 1 #self.true_rul[self.k] #/ (self.upper_bound[self.k] - self.lower_bound[self.k])
            
        #normalize on the length of the last 25% of the true RUL
        cic_25 = cic_25 / (len(self.true_rul) - int(len(self.true_rul) * 0.75))
        return cic_25
    
    def compute_cre(self):
        # Initialize CRE to 0
        cre = 0
        # Iterate over all time points
        for self.k in range(len(self.true_rul)-1):
            # Compute the relative error for the current time point and add it to CRE
            # if self.mean_rul[k] is empty, skip the iteration
            if np.isnan(self.mean_rul[self.k]):
                continue
            cre_now = (abs(self.true_rul[self.k]-self.mean_rul[self.k]) )/ self.true_rul[self.k]
            if cre_now > 1:
                cre_now = 1
            cre += 1 - cre_now
            # print((abs(self.true_rul[self.k]-self.mean_rul[self.k]))/ self.true_rul[self.k])
        cre = cre / self.true_rul[0]
        return cre
    
    def compute_cre_25(self):
        # Initialize CRE to 0
        cre_25 = 0
        # Iterate over all time points
        for k in range(int(len(self.true_rul) * 0.75), len(self.true_rul)-1):
            # Compute the relative error for the current time point and add it to CRE
            #if self.mean_rul[k] is nan, skip the iteration
            if np.isnan(self.mean_rul[k]):
                continue
            cre_now = (abs(self.true_rul[k]-self.mean_rul[k]) )/ self.true_rul[k]
            if cre_now > 1:
                cre_now = 1
            cre_25 += 1 - cre_now
            # print((abs(self.true_rul[self.k]-self.mean_rul[self.k]))/ self.true_rul[self.k])
        cre_25 = cre_25 / ((len(self.true_rul)-1) - int(len(self.true_rul) * 0.75))
        return cre_25
    
    def compute_beta(self):
        """
        Compute the beta metric.

        Parameters:
        - true_RUL: True Remaining Useful Life array, where true_RUL[0] = T_fail and true_RUL[T_fail] = 0.
        - alpha: Percentage for the bounds (e.g., 20% as 0.2).
        - predicted_RUL: Predicted Remaining Useful Life array.
        - lower_bound: Lower bounds of the predicted RUL.
        - upper_bound: Upper bounds of the predicted RUL.

        Returns:
        - beta: Beta metric value.
        """
        # Calculate the alpha bounds for true_RUL
        true_up_b = self.true_rul * (1 + self.alpha)
        true_low_b = self.true_rul * (1 - self.alpha)

        # Initialize the beta metric value
        beta = 0.0
        intersect_upper = np.zeros(len(self.true_rul))
        intersect_lower = np.zeros(len(self.true_rul))
        # Iterate through each time step
        for self.k in range(len(self.true_rul)-1):
            # Calculate the intersection of predicted bounds with true bounds
            intersect_upper[self.k] = min(self.upper_bound[self.k], true_up_b[self.k])
            intersect_lower[self.k] = max(self.lower_bound[self.k], true_low_b[self.k])
            # Check if there is a valid intersection
            if intersect_upper[self.k] > intersect_lower[self.k]:
                area = intersect_upper[self.k] - intersect_lower[self.k]

                norm_upper = intersect_upper[self.k] if intersect_upper[self.k] < true_up_b[self.k] else self.upper_bound[self.k]
                norm_lower = intersect_lower[self.k] if intersect_lower[self.k] > true_low_b[self.k] else self.lower_bound[self.k]
                normalization = 1/(norm_upper - norm_lower)
                
                beta += area*normalization
                
            else:
                intersect_lower[self.k] = np.nan
                intersect_upper[self.k] = np.nan

        beta = beta/self.true_rul[0]

        return beta
    
    def compute_beta_25(self):
        """
        Compute the beta metric.

        Parameters:
        - true_RUL: True Remaining Useful Life array, where true_RUL[0] = T_fail and true_RUL[T_fail] = 0.
        - alpha: Percentage for the bounds (e.g., 20% as 0.2).
        - predicted_RUL: Predicted Remaining Useful Life array.
        - lower_bound: Lower bounds of the predicted RUL.
        - upper_bound: Upper bounds of the predicted RUL.

        Returns:
        - beta: Beta metric value.
        """
        # Calculate the alpha bounds for true_RUL
        true_up_b = self.true_rul * (1 + self.alpha)
        true_low_b = self.true_rul * (1 - self.alpha)

        # Initialize the beta metric value
        beta_25 = 0.0
        intersect_upper = np.zeros(len(self.true_rul))
        intersect_lower = np.zeros(len(self.true_rul))
        # Iterate through each time step
        for k in range(int(len(self.true_rul) * 0.75), len(self.true_rul)-1):
            # Calculate the intersection of predicted bounds with true bounds
            intersect_upper[k] = min(self.upper_bound[k], true_up_b[k])
            intersect_lower[k] = max(self.lower_bound[k], true_low_b[k])

            # Check if there is a valid intersection
            if intersect_upper[k] >= intersect_lower[k]:
                area = intersect_upper[k] - intersect_lower[k]

                norm_upper = intersect_upper[k] if intersect_upper[k] < true_up_b[k] else self.upper_bound[k]
                norm_lower = intersect_lower[k] if intersect_lower[k] > true_low_b[k] else self.lower_bound[k]
                normalization = 1/(norm_upper - norm_lower)

                beta_25 += area*normalization
            else:
                intersect_lower[k] = np.nan
                intersect_upper[k] = np.nan

        beta_25 = beta_25/((len(self.true_rul)-1) - int(len(self.true_rul) * 0.75))
        return beta_25