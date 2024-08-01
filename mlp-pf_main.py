import hydra
import joblib
from omegaconf import DictConfig
from tqdm import tqdm
from mlp_pf_core import MLP_PF

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Accédez aux paramètres à partir du fichier de configuration

    # test_pack = cfg.exp.test_pack
    # training_pack = cfg.exp.training_pack
    # dir_project = cfg.dir_project
    # model_name = cfg.exp.model_name
    # var_x = cfg.exp.var_x
    # var_y = cfg.exp.var_y
    # degradation = cfg.exp.degradation
    # y_failure = cfg.exp.y_failure
    # horizon = cfg.exp.horizon

    # save_data_fig = cfg.exp.save_data_fig
    # show_fig = cfg.exp.show_fig
    # save_weights = cfg.exp.save_weights
    # load_weights = cfg.exp.load_weights

    # Ns = cfg.exp.Ns
    # hidden_nodes = cfg.exp.hidden_nodes
    # epochs = cfg.exp.epochs

    # s2x0 = cfg.exp.s2x0
    # s2x1 = cfg.exp.s2x1
    # s2x2 = cfg.exp.s2x2
    # s2z     = cfg.exp.s2z

    # alpha = cfg.exp.alpha
    # random_seed = cfg.exp.random_seed
    
    # Create and run the particle filter
    battery_pf = MLP_PF(dir_project   = cfg.dir_project,
                        test_pack     = cfg.exp.test_pack, 
                        training_pack = cfg.exp.training_pack, 
                        model_name    = cfg.exp.model_name,
                        var_x         = cfg.exp.var_x,
                        var_y         = cfg.exp.var_y,
                        degradation   = cfg.exp.degradation,
                        horizon       = cfg.exp.horizon,
                        y_failure     = cfg.exp.y_failure,
                        save_data_fig = cfg.exp.save_data_fig, 
                        show_fig      = cfg.exp.show_fig, 
                        save_weights  = cfg.exp.save_weights, 
                        load_weights  = cfg.exp.load_weights, 
                        Ns            = cfg.exp.Ns, 
                        hidden_nodes  = cfg.exp.hidden_nodes,
                        epochs        = cfg.exp.epochs, 
                        s2z           = cfg.exp.s2z, 
                        s2x0          = cfg.exp.s2x0, 
                        s2x1          = cfg.exp.s2x1, 
                        s2x2          = cfg.exp.s2x2,
                        alpha         = cfg.exp.alpha,
                        random_seed   = cfg.exp.random_seed
                        )
        


    (cic, cre, beta,
     cic_25, cre_25, beta_25 ) = battery_pf.run()

    # RUN HERE YOUR exp
    # python main_hydra.py will use base with default exp
    # "python main_hydra.py exp=fcg" to run exp fcg
    # same with vhi or any other exp you create
    # python main_hydra.py --multirun experiment=fcg,experiment=vhi

    # python3 main_hydra.py exp=fcg exp.s2x0=0.05
    
    
if __name__ == "__main__":
    main()
