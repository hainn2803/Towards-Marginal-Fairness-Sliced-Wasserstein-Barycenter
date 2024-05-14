import numpy as np
import json

def calculate_mean_std(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    # mean_format = "{:.{}f}".format(mean, precision)
    # std_dev_format = "{:.{}f}".format(std_dev, precision)
    return mean, std_dev


def extract_values(log_path):
    # Initialize lists to store values
    WG_list = []
    LP_list = []
    RL_values = []
    F_latent_list = []
    W_latent_list = []

    with open(log_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Check if the line contains the desired values
            
            if '+) Wasserstein distance between generated and real images:' in line:
                WG_list.append(float(line.split(':')[-1].strip()))
            elif '+) Wasserstein distance between posterior and prior distribution' in line:
                LP_list.append(float(line.split(':')[-1].strip()))
            elif '+) Reconstruction loss:' in line:
                RL_values.append(float(line.split(':')[-1].strip()))
            elif '+) Fairness:' in line:
                F_latent_list.append(float(line.split(':')[-1].strip()))
            elif '+) Averaging distance:' in line:
                W_latent_list.append(float(line.split(':')[-1].strip()))


    return WG_list, LP_list, RL_values, F_latent_list, W_latent_list
# Example usage:

if __name__ == "__main__":
    
    list_FSW_value = ["0.0"]
    CKP = "300"
    
    all_latex_codes = ""
    template_file_path = "table_template.txt"
    
    for FSW_value in list_FSW_value:
    
        RESULT = ["result"]
        DATASET = ["mnist"]
        SEED = ["seed_28"]
        LR = ["lr_0.001"]
        FSW = [f"fsw_{FSW_value}"]
        METHOD = ["EFBSW", "FBSW", "lowerboundFBSW", "OBSW_10.0","OBSW_1.0", "OBSW_0.1", "BSW"]
        METHOD_NAME = ["es-MFSWB", "us-MFSWB", "s-MFSWB", "MFSWB $\lambda = 10.0$", "MFSWB $\lambda = 1.0$", "MFSWB $\lambda = 0.1$", "USWB"]
        METHOD = ["None"]
        METHOD_NAME = ["None"]
        # METHOD = ["OBSW_10.0","OBSW_1.0", "OBSW_0.1"]
        # METHOD_NAME = ["MFSWB $\lambda = 10.0$", "MFSWB $\lambda = 1.0$", "MFSWB $\lambda = 0.1$"]
        # METHOD = ["EFBSW", "FBSW", "lowerboundFBSW", "BSW"]
        # METHOD_NAME = ["es-MFSWB", "us-MFSWB", "s-MFSWB", "USWB"]
        DATA_value = ""
        for r in RESULT:
            for d in DATASET:
                for s in SEED:
                    for l in LR:
                        res_latex = ""
                        for i in range(len(METHOD)-1, -1, -1):
                            res_method_latex = f"{METHOD_NAME[i]}"
                            for f in FSW:
                                m = METHOD[i]
                                log_path = f"{r}/{d}/{s}/{l}/{f}/{m}/evaluate_epoch_{CKP}_{m}.log"
                                WG_list, LP_list, RL_values, F_latent_list, W_latent_list = extract_values(log_path)
                                
                                print(f"Method: {m}, number of sample testing: {len(RL_values)}, {len(set(RL_values))}")

                                mean_LP, std_LP = calculate_mean_std(LP_list)
                                mean_WG, std_WG = calculate_mean_std(WG_list)
                                mean_F_latent, std_F_latent = calculate_mean_std(F_latent_list)
                                mean_W_latent, std_W_latent = calculate_mean_std(W_latent_list)
                                
                                mean_LP *= 10**2
                                mean_F_latent *= 10**2
                                mean_W_latent *= 10**2

                                # assert len(set(RL_values)) == 1
                                res_method_latex += " & {:.2f}".format(RL_values[-1]) \
                                    + "& {:.2f}".format(mean_LP) \
                                        + "& {:.2f}".format(mean_WG) \
                                            + "& {:.2f}".format(mean_F_latent) \
                                                + "& {:.2f}".format(mean_W_latent)
                 
                            
                            res_method_latex += "\\\\"
                            res_method_latex += "\n"
                            DATA_value += res_method_latex

        with open(template_file_path, "r") as template_file:
            latex_template = template_file.read()
            
            latex_code = latex_template.replace("@FSW", FSW_value)
            latex_code = latex_code.replace("@EPOCHS", CKP)
            latex_code = latex_code.replace("@DATA", DATA_value)
        
        all_latex_codes += latex_code + "\n"

        output_file_path = "output.txt"
        with open(output_file_path, "w") as output_file:
            output_file.write(all_latex_codes)

        print("Updated LaTeX code has been written to:", output_file_path)