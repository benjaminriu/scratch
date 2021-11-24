if True:#For code folding
    #For a NVIDIA 2080Ti (11.G of VRAM):
    MAX_GPU_MATRIX_WIDTH = int(4096) #Max width of any matrix; scales quadraticaly with VRAM
    MAX_GPU_NETWORK_DEPTH = int(6) #Max depth of any network; scales linearly with VRAM
    #For same VRAM size, you can pick a different tradeoff between those two.
    #You could also increase matrix width or network depth beyond these values if you reduce either your batch size or network width accordingly

    import numpy as np
    import pandas as pd
    import sklearn as sk
    import torch as torch

    from sklearn.metrics import r2_score as r2
    from sklearn.metrics import accuracy_score as acc
    from sklearn.model_selection import train_test_split as tts

    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.base import is_classifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
    from scipy.special import expit as logistic_func
    from abc import ABCMeta, abstractmethod
    import time

    from sklearn.utils import check_random_state
    from sklearn.utils import shuffle

    #USE GPU
    CUDA = True
    if torch.cuda.is_available() and CUDA: dev = "cuda:0"
    else: dev = "cpu"
    device = torch.device(dev)

    #quick shortcuts
    def n2t(array):
        #numpy array to torch Tensor
        if type(array) == torch.Tensor: return array
        return torch.tensor(array).to(device).float()
    def t2n(tensor):
        #torch Tensor to numpy array
        if type(tensor) == np.ndarray: return tensor
        return tensor.detach().to("cpu").numpy()
    def n2f(array):
        #numpy array to float
        if type(array) == torch.Tensor: array = t2n(array)
        if type(array) == float: return array
        else: return np.array(array).reshape(-1)[0]
    def f2s(value, length = 8, delimiter = " |"):
        #float to string with fixed length
        if type(value) == str:
            return " " * (length - len(value)) + value[:length] + delimiter
        else:
            return ("%." +str(length-6)+"e") % float(value) + delimiter

    def no_context():
        import contextlib
        @contextlib.contextmanager
        def dummy_context_mgr():
            yield None
        return dummy_context_mgr


    PATH = "tempory_model.pt"#see _save_weights and _load_weights
    GRID_START, GRID_END, GRID_SIZE = 1e1, 1e7, 101 #see _gridsearch_closeform_param
    MAX_TRAINING_SAMPLES, MAX_TREE_SIZE = int(1e4), 50 #see _stratify_continuous_target (CPU memory and runtime safeguards)
    from sklearn.preprocessing import LabelEncoder
    from scipy.special import softmax as softmax_func
    from scipy.special import logit as logit_func

class MLRNet(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, #see subclasses for default values
        hidden_nn,#architecture of hidden layers, torch.nn.Module
        hidden_params,#hyper_parameters for hidden_nn initialization
                 
        #learning parameters
        default_loss_function, #set by the subclass
        optimizer, #a torch optimizer, str
        learning_rate, #float
        optimizer_params,#except learning rate, dict
        lr_scheduler, #a torch optimizer.lr_scheduler, str
        lr_scheduler_params,#dict
        batch_size, #if None or False: full batch, if int number of samples, if float share of samples
                 
        #convergence parameters
        max_iter, #iterations, not epochs (epochs = max_iter/batch_size), int
        max_runtime, #unprecise, float or int
        validation_fraction, #if None or False: no validation, if int number of samples, if float share of samples
        should_stratify, #validation split strategy, bool
        early_stopping_criterion, #either "loss" or "validation", str
        convergence_tol, #if None or False: always max_iter, else float 
        divergence_tol, #if None or False: always max_iter, else float 
        
        #MuddlingLabelRegularization specific parameters
        closeform_parameter_init, #if None or False: regular FFNN, if int or float lambda initial value, if "max_variation" or "min_value" grid-search
        closeform_intercept,#add unitary feature to covar matrix, bool
        n_permut, #if int number of permutations, if None or False no permutations
        permutation_scale, #weight of permutation term added to the loss, float
        dithering_scale,#if float dithering white noise standard-deviation, if None or False no gaussian dithering
        target_rotation_scale,#if float dithering structured noise standard-deviation, if None or False no structured noise dithering
        
        center_target,#center target around mean (behaves differently for binary clf), bool 
        rescale_target,#divide target by std before fitting, bool 
        loss_imbalance,#smaller weights on majority classes, bool
        random_state, #scikit-learn random state, will also set torch generator using a different seed
        verbose #if False mute, if True print at each iteration, if int print if iter%verbose == 0
                ):
        
        self.hidden_nn = hidden_nn
        self.hidden_params = hidden_params
        self.default_loss_function = default_loss_function
        self.optimizer  = optimizer
        self.learning_rate  = learning_rate
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params 
        
        self.batch_size  = batch_size
        self.max_iter  = max_iter
        self.max_runtime  = max_runtime
        self.validation_fraction  = validation_fraction
        self.should_stratify = should_stratify
        self.early_stopping_criterion  = early_stopping_criterion
        self.convergence_tol  = convergence_tol
        self.divergence_tol  = divergence_tol
        self.closeform_parameter_init  = closeform_parameter_init
        self.closeform_intercept = closeform_intercept
        self.n_permut  = n_permut
        self.permutation_scale = permutation_scale
        self.dithering_scale  = dithering_scale
        self.target_rotation_scale = target_rotation_scale
        
        self.center_target = center_target
        self.rescale_target = rescale_target
        self.loss_imbalance = loss_imbalance
        self.random_state  = random_state
        self.verbose  = verbose
        
    def fit(self, X, y):
        return self._fit(X,y, incremental=False)
    
    def partial_fit(self, X, y):
        return self._fit(X, y, incremental=True) 
    
    def initialize_model(self, X, y):
        self.init_time = time.time()
        self._initialize_state()
        y = self._initialize_task(y)
        if self.validation: X, self.X_valid, y, self.y_valid = self._initialize_valid(X, y)
        self._initialize_training(len(y))
        datas = self._initialize_data(X,y)
        if self.use_closeform: self._initialize_closeform(datas)
        self.init_time = time.time() - self.init_time
        return datas
        
    def _fit(self, X, y, incremental=False):
        if incremental:
            datas = self._initialize_data(X,y)
        else:
            datas = self.initialize_model(X,y)
        self._train_model(datas)
        return self
    
    def _train_model(self, datas):
        self.current_time = time.time()
        #run until termination
        while self.run_one_iter(datas):
            datas = self._update_data(datas)
            
        #end training
        if self.early_stopping: self._load_weights()
        self._release_train_memory()
    
    def run_one_iter(self, datas):
        loss = self._forward_propagate(datas)
        if self._check_termination():
            self.has_converged = True
        else:
            self._backward_propagate(loss)
        del loss
        return not self.has_converged

    def _initialize_state(self):
        self.init_time = time.time()
        self._random_state = check_random_state(self.random_state)
        self.torch_random_state = torch.Generator(device=device).manual_seed(int(self._random_state.uniform(0,2**31)))
        self.print_record = self.verbose not in [0, False]
        self.validation = self.validation_fraction not in [False, None]
        self.check_convergence = self.convergence_tol not in [False, None]
        self.check_divergence = self.divergence_tol not in [False, None]
        self.early_stopping = self.early_stopping_criterion not in [False, None]
        self.minimize_criterion = self.early_stopping_criterion != "validation"
        self.use_closeform = self.closeform_parameter_init not in [0, 0., False, None]
        self.add_permut_term = self.use_closeform and self.n_permut not in [False, None, 0, 0.] and type(self.permutation_scale) not in [type(False), type(None)]
        self.diff_permut_term = self.add_permut_term and self.permutation_scale not in [False, None, 0, 0.]
        self.add_target_rotation = self.target_rotation_scale not in [False, None, 0, 0.] 
        self.add_dithering = self.dithering_scale not in [False, None, 0, 0.] 
        self._initialize_record()
        
    def _initialize_record(self):
        self.record = {"loss":[],"time":[]}
        if self.validation:
            self.record["validation"] = []
        if self.use_closeform:
            self.record["lambda"] = []
        if self.add_permut_term:
            self.record["mlr"] = []
            
    def _initialize_task(self, y):
        n_classes = len(np.unique(y)) if is_classifier(self) else 0
        self.multi_class = n_classes >2
        self.class_mean = 1./n_classes if n_classes else 0.
        
        if self.multi_class:
            label_encoder = LabelEncoder().fit(y)
            y = label_encoder.transform(y)
            class_count = np.unique(y, return_counts=True)[1]
            class_share = class_count/class_count.sum()
            self._set_multiclass(label_encoder, n_classes, class_share)
        elif n_classes == 2:
            pos_share = y.mean()
            self._set_binaryclass(pos_share)
        else:
            target_std = float(y.std()) if self.rescale_target else 1.
            static_intercept = y.mean()/target_std if self.center_target else 0.
            self._set_continuous(target_std, static_intercept)
        return y    
    
    def _set_continuous(self, target_std = 1.0, static_intercept = 0.):
        self.target_type = "continuous"
        self.n_classes = 0
        self.class_mean = 0
        self.target_std = target_std
        self.static_intercept = static_intercept
        if self.loss_imbalance:
            self.classweights = {}
            
    def _set_binaryclass(self, pos_share = 0.5):
        self.target_type = "binaryclass"
        self.n_classes = 2
        self.class_mean = 0.5
        self.pos_share = pos_share
        if self.loss_imbalance:
            pos_weight = (1 - self.pos_share) / self.pos_share
            self.classweights = {"pos_weight":n2t(pos_weight)}
        if self.center_target:
            self.static_intercept = logit_func(self.pos_share)
        
    def _set_multiclass(self, label_encoder, n_classes, class_share = None, static_intercept = 0.):
        self.target_type = "multiclass"
        self.n_classes = n_classes
        self.class_mean = 1./n_classes
        self.label_encoder = label_encoder
        self.class_share = class_share
        if self.loss_imbalance:
            class_weights = self.class_share.max() / self.class_share
            class_weights = class_weights / class_weights.sum()
            self.classweights = {"weight" : n2t(class_weights)}
        if self.center_target:
            self.static_intercept = static_intercept

    def _initialize_training(self, data_length):
        self.batch_learning = type(self.batch_size) in [float, int] or (data_length > MAX_GPU_MATRIX_WIDTH and device == "cuda")
        if type(self.batch_size) in [float] and not self.batch_learning:
            if self.batch_size * data_length > MAX_GPU_MATRIX_WIDTH and device == "cuda": self.batch_learning = True
        loss_params = self.classweights if self.loss_imbalance else {} 
        self.loss_params = {}
        self.loss_function = self.default_loss_function if not self.multi_class else "CrossEntropyLoss"
        self.loss_func = getattr(torch.nn,self.loss_function)(reduction='none', **loss_params)
        if device != "cpu":
            torch.cuda.manual_seed_all(int(self._random_state.uniform(0,2**31))) #Otherwise weights initialized without fix seed
            self.hidden_layers = self.hidden_nn(**self.hidden_params)
        else:
            self.hidden_layers = self.hidden_nn(**self.hidden_params)
        params = list(self.hidden_layers.parameters())
        if self.use_closeform:
            self.closeform_param = torch.nn.Parameter(n2t(np.zeros(1)))
            params += [self.closeform_param]
            self.output_weights = None
        self.optimizer_instance  = getattr(torch.optim, self.optimizer)(lr = self.learning_rate, params = params, **self.optimizer_params)
        self.cst_lr = self.lr_scheduler in [None,False]
        if not self.cst_lr:
            self.update_lr_every_step = (self.batch_learning == False) or self.lr_scheduler in ["OneCycleLR","CyclicLR"]
            self.lr_scheduler_instance = getattr(torch.optim.lr_scheduler, self.lr_scheduler)(self.optimizer_instance, **self.lr_scheduler_params)
        del params                
        self.current_iter = 0
        self.has_converged = False            
   
    def _initialize_valid(self, X, y):
        def roc_with_proba(y, probas): return roc_auc_score(y, probas[:,-1])
        def roc_with_proba_ovr(y, probas): return roc_auc_score(y, probas, multi_class ="ovr")
        if is_classifier(self):
            if self.multi_class:
                self.valid_metric = roc_with_proba_ovr
            else:
                self.valid_metric = roc_with_proba
        else:
            self.valid_metric = r2_score
        self.valid_func = self.predict_proba if is_classifier(self) else self.predict
        self.validation = True
        if self.should_stratify:
            if is_classifier(self): stratify = y 
            else: 
                self.target_std
                stratify = self._stratify_continuous_target(y/self.target_std)
        else: stratify = None
        X, X_valid, y, y_valid = train_test_split(
            X, y, random_state=self._random_state,
            test_size=self.validation_fraction,
            stratify=stratify)
        return X, n2t(X_valid), y, y_valid
        
    def _stratify_continuous_target(self, y):
        from sklearn.tree import DecisionTreeRegressor as tree_binarizer
        MAX_TRAINING_SAMPLES, MAX_TREE_SIZE = int(1e4), 50 #CPU memory and runtime safeguards
        tree_binarizer_params = {"criterion":'friedman_mse', 
                   "splitter":'best', 
                   "max_depth":None, 
                   "min_samples_split":2, 
                   "min_weight_fraction_leaf":0.0, 
                   "max_features":None,  
                   "min_impurity_decrease":0.2, 
                   "min_impurity_split":None, 
                   "ccp_alpha":0.0}
        
        tree_size = min(int(np.sqrt(len(y))), MAX_TREE_SIZE)
        fit_sample_size = min(len(y), MAX_TRAINING_SAMPLES)
        tree_binarizer_params["random_state"] = self._random_state
        tree_binarizer_params["max_leaf_nodes"] = tree_size
        tree_binarizer_params["min_samples_leaf"] = tree_size
        return tree_binarizer(**tree_binarizer_params).fit(y[:fit_sample_size].reshape((-1,1)), y[:fit_sample_size]).apply(y.reshape((-1,1))) 
   
    def _initialize_batch(self, n_samples):
        if type(self.batch_size) == float:
            self.batch_length = min(MAX_GPU_MATRIX_WIDTH, int(n_samples * self.batch_size))
        elif type(self.batch_size) == int:
            self.batch_length = min(MAX_GPU_MATRIX_WIDTH, self.batch_size, n_samples)
        elif n_samples > MAX_GPU_MATRIX_WIDTH:
            self.batch_length = int(MAX_GPU_MATRIX_WIDTH)
        self.n_batches = int(n_samples / self.batch_length)
        
    def _initialize_data(self, X, y):
        if self.batch_learning:
            self._initialize_batch(len(y))
        if not self.multi_class:
            y = y.reshape((X.shape[0],-1)).T
        if self.rescale_target:
            y = y / self.target_std
        if self.center_target and not is_classifier(self): 
            y = y - self.static_intercept
        datas = {"X":X, "y":y}
        if self.add_permut_term:
            datas["y_permuted"] = self._permut_label(y)
        if self.batch_learning:
            self._generate_batch(datas["X"].shape[0])
            datas = self._update_data(datas)
        else: 
            datas["input"] = n2t(datas["X"])
            datas["target"] = self._cast(n2t(datas["y"]))
            if self.add_permut_term: datas["target_permuted"] = self._cast(n2t(datas["y_permuted"]))
        return datas
            
    def _update_data(self, datas):
        if self.batch_learning:
            if len(self.batches) == 0:
                self._generate_batch(datas["X"].shape[0])
            batch_indexes = self.batches.pop(0)
            datas["input"] = n2t(datas["X"][batch_indexes])
            datas["target"] = self._cast(n2t(datas["y"][..., batch_indexes]))
            if self.add_permut_term: datas["target_permuted"] = self._cast(n2t(datas["y_permuted"][..., batch_indexes]))
        return datas
   
    def _permut_label(self, y):
        y_index = np.arange(y.shape[-1])
        return np.concatenate([y[np.newaxis,..., shuffle(y_index,random_state=self._random_state)] for permut in range(self.n_permut)], axis=0)

    def _generate_batch(self, n_samples):
        shuffled_indexes = shuffle(np.arange(n_samples),random_state=self._random_state)
        self.batches = [shuffled_indexes[batch_i * self.batch_length: (batch_i + 1) * self.batch_length] for batch_i in range(self.n_batches)]

    def _initialize_closeform(self, datas):
        if self.add_permut_term: 
            with torch.no_grad():
                target = datas["target"]
                formated_target = self._format_target(target)
                pred = formated_target.mean() * torch.ones(formated_target.shape, device = device)
                self.intercept = self._compute_loss(pred, target).detach() 
        if type(self.closeform_parameter_init) in [float, int] or isinstance(self.closeform_parameter_init, np.number):
            closeform_param =  self.closeform_parameter_init
        elif self.closeform_parameter_init in ["min_value", "max_variation"]:
            closeform_param = self._gridsearch_closeform_param(datas)
        else:
            closeform_param = 1.
        with torch.no_grad():
            self.closeform_param += n2t(np.log(closeform_param))   
                
    def _gridsearch_closeform_param(self, datas):
        candidates, losses = np.geomspace(GRID_START, GRID_END, GRID_SIZE), np.zeros(GRID_SIZE)
        with torch.no_grad():
            activation = self._forward_pass(datas["input"])
            if self.closeform_intercept: activation = torch.cat([activation, torch.ones([activation.shape[0], 1], device = device)], dim = -1)
            target = self._dither_target(datas["target"])
            formated_target = self._format_target(target)
            target_dot_activation =  formated_target @ activation
            target_permuted_dot_activation = None
            if self.diff_permut_term:
                target_permuted = self._dither_target(datas["target_permuted"])
                formated_target_permuted = self._format_target(target_permuted)
                target_permuted_dot_activation = formated_target_permuted @ activation
            for i,candidate in enumerate(candidates):
                inv_mat_dot_activation = self._get_inv_mat(activation, n2t(np.log(candidate))) @ activation.T
                pred = target_dot_activation @ inv_mat_dot_activation
                loss = self._compute_loss(pred, target)
                if self.diff_permut_term:
                    pred_permut = target_permuted_dot_activation @ inv_mat_dot_activation
                    loss += self.permutation_scale * self._compute_MLR_penalty(pred_permut, target_permuted)
                losses[i] = n2f(t2n(loss))
        self.init_losses = losses
        if self.closeform_parameter_init == "max_variation":
            return np.geomspace(GRID_START, GRID_END, GRID_SIZE-1)[np.argmax(losses[1:] - losses[:-1])]
        else:
            return candidates[np.argmin(losses)]
                   
    def _valid_score(self):
        return self.valid_metric(self.y_valid, self.valid_func(self.X_valid))              
    
    def _reduce_loss(self, point_wise_loss):
        if self.loss_function == "MSELoss":
            return torch.sqrt(point_wise_loss.mean(dim = -1))
        else: return point_wise_loss.mean(dim = -1)

    def _get_inv_mat(self, activation, ridge_coef):
        correl_mat = activation.transpose(1,0) @ activation
        #With gpu, torch.inverse can fail for no valid reason (equivalent operation works on cpu and numpy)
        #eg.: RuntimeError: inverse_cuda: For batch 0: U($w,$w) is zero, singular U.
        #This is caused by the MAGMA subroutine, matrix is not actually singular
        #see https://github.com/pytorch/pytorch/issues/29096 and therein
        #Safest way to prevent this is to use larger diagonal values (ie ridge_coef *= 10)
        inversion_successful = False
        inversion_attempts = 0
        MAX_ATTEMPTS = 10
        while not inversion_successful and inversion_attempts < MAX_ATTEMPTS:
            try:
                diag_mat = torch.diag(torch.ones(activation.shape[1],device = device) * torch.exp(ridge_coef))
                inversed_mat = torch.inverse(correl_mat + diag_mat)
                inversion_successful = True
            except:
                print("Inversion failed, attempt:", str(inversion_attempts))
                ridge_coef += n2t(np.log(10.))#multiply diag by 10 until inversion works
                inversion_attempts += 1
        if inversion_successful:
            return inversed_mat
        else:
            print("All inversion attempts failed")
            self.failed_inversion = True
            return torch.eye(correl_mat.shape, device = device)#In that case pred = X @ X.T @ Y
      
    def _compute_loss(self, pred, target):
        if self.multi_class:
            pred = pred.T
        return self._reduce_loss(self.loss_func(pred, target)).mean()
        
    def _compute_MLR_penalty(self, pred, target):
        return torch.abs(self.intercept - self._reduce_loss(self.loss_func(pred, target))).mean()
        
    def _forward_pass(self, activation):
        return self.hidden_layers.train().forward(activation) 

    def _backward_propagate(self, loss):
        loss.backward()
        self.optimizer_instance.step()
        if not self.cst_lr:
            if self.update_lr_every_step:
                self.lr_scheduler_instance.step()          
            elif len(self.batches) == 0:
                self.lr_scheduler_instance.step()
        self.optimizer_instance.zero_grad()
        self.current_iter = self.current_iter + 1
        
    def _cast(self, target):
        if self.multi_class:
            return target.long()
        else:
            return target
     
    def _format_target(self, target):
        if self.multi_class:
            target = torch.transpose(torch.nn.functional.one_hot(target, num_classes = self.n_classes),-1,-2)
        if is_classifier(self):#center classes
            target = (target - self.class_mean)*2
        return target
    
    def _rotate_pred(self, target, projector):
        if self.add_target_rotation:
            epsilon = torch.normal(0.,self.target_rotation_scale, size = target.shape ,generator = self.torch_random_state, device = device)
            target = (target + target @ projector)/2 +  epsilon - epsilon @ projector
        return target

    def _dither_target(self, target):
        if self.add_dithering:
            target = target + torch.normal(0., self.dithering_scale, size = target.shape, generator = self.torch_random_state, device = device)
        return target
        
    def _forward_propagate(self, datas):
        activation, target = datas["input"], datas["target"]
        target = self._dither_target(target)
        activation = self._forward_pass(activation)
        if self.use_closeform:
            self.record["lambda"].append(np.exp(n2f(t2n(self.closeform_param))))
            formated_target = self._format_target(target)
            if self.closeform_intercept: activation = torch.cat([activation, torch.ones([activation.shape[0], 1], device = device)], dim = -1)
            inv_mat = self._get_inv_mat(activation, self.closeform_param)
            activation_dot_inv_mat = activation @ inv_mat
            beta = formated_target @ activation_dot_inv_mat
            self.output_weights = beta

            if self.add_target_rotation or self.add_permut_term:
                projector = activation_dot_inv_mat @ activation.T
            if self.add_target_rotation:
                pred = self._rotate_pred(formated_target, projector)
            else:
                pred = beta @ activation.T
        else:
            pred = activation.T 
            if self.center_target:
                pred += self.static_intercept
        loss = self._compute_loss(pred, target)
        self.record["loss"].append(n2f(t2n(loss)))
        
        if self.add_permut_term:
            with no_context()() if self.diff_permut_term else torch.no_grad():
                target_permutation = datas["target_permuted"]
                target_permutation = self._dither_target(target_permutation)
                formated_target_permutation = self._format_target(target_permutation)
                if self.add_target_rotation:
                    pred_permutation = self._rotate_pred(formated_target_permutation, projector)
                else:
                    pred_permutation = formated_target_permutation @ projector
                permut_loss = self._compute_MLR_penalty(pred_permutation, target_permutation)
                self.record["mlr"].append(n2f(t2n(permut_loss)))
                if self.diff_permut_term:
                    loss += self.permutation_scale * permut_loss
            
        if self.validation: self.record["validation"].append(self._valid_score())
        if self.early_stopping: self._save_weights()
        self.record["time"].append(time.time() - self.current_time)
        self.current_time = time.time()
        if self.print_record: 
            if self.current_iter == 0:
                print("| "+f2s("iter"), *map(f2s, self.record.keys()))
            if self.current_iter % int(self.verbose) == 0:
                print("| "+f2s(str(self.current_iter)), *map(lambda value : f2s(value[-1]), self.record.values()))  
        return loss
    
    def _check_termination(self):
        if self.current_iter == 0:
            return False
        else:
            return self._check_convergence() or self._check_divergence() or self._check_timeout() or self.current_iter >= self.max_iter
    def _check_convergence(self):
        if self.check_convergence:
            return np.abs(np.min(self.record["loss"][:-1]) - self.record["loss"][-1]) < self.convergence_tol
        else: return False
    def _check_divergence(self): 
        if self.check_divergence:
            return self.record["loss"][-1] > self.divergence_tol
        else: return False
    def _check_timeout(self):
        return self.max_runtime < self.init_time + np.sum(self.record["time"])
    
    def _save_weights(self):
        if self.current_iter == 0:
            self.best_iter = self.current_iter
        elif bool(self.record[self.early_stopping_criterion][-1]  < self.record[self.early_stopping_criterion][self.best_iter]) == self.minimize_criterion:
            self.best_iter = self.current_iter
        if self.current_iter == self.best_iter:
            torch.save(self.hidden_layers, PATH)
            if self.use_closeform: self.saved_output_weights = torch.clone(self.output_weights)
            
    def _load_weights(self):
        del self.hidden_layers
        self.hidden_layers = torch.load(PATH)
        if self.use_closeform:
            del self.output_weights
            self.output_weights = self.saved_output_weights
        
    def _release_train_memory(self):
        if self.validation: del self.X_valid, self.y_valid
        if self.use_closeform: del self.closeform_param
        del self.optimizer_instance
        torch.cuda.empty_cache()
        
    def _forward_pass_fast(self, activation):
        with torch.no_grad():
            hidden_output = self.hidden_layers.eval().forward(activation)
            if self.use_closeform:
                if self.closeform_intercept: hidden_output = torch.cat([hidden_output, torch.ones([hidden_output.shape[0], 1], device = device)], dim = -1)
                output = (self.output_weights @ hidden_output.T).T
            else: output = hidden_output
            if self.center_target and not (is_classifier(self) and self.use_closeform):
                output += self.static_intercept
            if self.rescale_target:
                output = output * self.target_std
        return output
    
    def _predict_hidden(self, X):
        if X.shape[0] <= MAX_GPU_MATRIX_WIDTH:
            return t2n(self._forward_pass_fast(n2t(X)))
        else:
            return np.concatenate([ self._predict_hidden(X[:MAX_GPU_MATRIX_WIDTH]), self._predict_hidden(X[MAX_GPU_MATRIX_WIDTH:])])
 
    def delete_model_weights(self):
        del self.hidden_layers
        if self.use_closeform: del self.output_weights
        torch.cuda.empty_cache()        
        
class MLRNetRegressor(RegressorMixin, MLRNet):
    def __init__(self,
        hidden_nn,
                *,
        hidden_params = {},
        optimizer = "Adam",
        lr_scheduler = False,
        learning_rate = 1e-2,
        optimizer_params = {},
        lr_scheduler_params = {},   
        batch_size = False,
        max_iter = 50,
        max_runtime = 300,       
        validation_fraction = 0.2,
        should_stratify = True,
        early_stopping_criterion = "validation",
        convergence_tol = False,
        divergence_tol = False,
        closeform_parameter_init = "max_variation",
        closeform_intercept = True,
        n_permut = 16,
        permutation_scale = 1.,
        dithering_scale = False,
        target_rotation_scale = False,
        center_target = False,
        rescale_target = True,
        loss_imbalance = False,
        random_state = None,
        verbose = False
                ):
        super().__init__(
        hidden_nn = hidden_nn, hidden_params = hidden_params,
        default_loss_function = 'MSELoss', optimizer  = optimizer, learning_rate  = learning_rate,optimizer_params = optimizer_params,
        lr_scheduler = lr_scheduler, lr_scheduler_params = lr_scheduler_params,
        batch_size  = batch_size, max_iter  = max_iter, max_runtime  = max_runtime, validation_fraction  = validation_fraction,
        should_stratify = should_stratify, early_stopping_criterion  = early_stopping_criterion, convergence_tol  = convergence_tol, divergence_tol  = divergence_tol,
        closeform_parameter_init  = closeform_parameter_init, closeform_intercept = closeform_intercept, n_permut  = n_permut, permutation_scale = permutation_scale, dithering_scale = dithering_scale,
        target_rotation_scale =  target_rotation_scale, center_target = center_target, rescale_target = rescale_target, loss_imbalance = loss_imbalance,
            random_state  = random_state, verbose  = verbose)
        
    def predict(self, X):
        return self._predict_hidden(X)
    
class MLRNetClassifier(ClassifierMixin, MLRNet):
    def __init__(self, 
        hidden_nn,
                *,
        hidden_params = {},
        optimizer = "Adam",
        lr_scheduler = False,
        learning_rate = 1e-2,
        optimizer_params = {},
        lr_scheduler_params = {},   
        batch_size = False,
        max_iter = 50,
        max_runtime = 300,       
        validation_fraction = 0.2,
        should_stratify = True,
        early_stopping_criterion = "validation",
        convergence_tol = False,
        divergence_tol = False,
        closeform_parameter_init = "max_variation",
        closeform_intercept= True,
        n_permut = 16,
        permutation_scale = 1.,
        dithering_scale = None,
        target_rotation_scale = False,
        center_target = False,
        rescale_target = False,
        loss_imbalance = False,
        random_state = None,
        verbose = False
                ):
        super().__init__(
        hidden_nn = hidden_nn, hidden_params = hidden_params,
        default_loss_function = 'BCEWithLogitsLoss', optimizer  = optimizer, learning_rate  = learning_rate, optimizer_params = optimizer_params,
        lr_scheduler = lr_scheduler, lr_scheduler_params = lr_scheduler_params,
        batch_size  = batch_size, max_iter  = max_iter, max_runtime  = max_runtime, validation_fraction  = validation_fraction,
        should_stratify= should_stratify, early_stopping_criterion  = early_stopping_criterion, convergence_tol  = convergence_tol, divergence_tol  = divergence_tol,
        closeform_parameter_init  = closeform_parameter_init, closeform_intercept = closeform_intercept, n_permut  = n_permut, permutation_scale = permutation_scale, dithering_scale = dithering_scale,
        target_rotation_scale =  target_rotation_scale, center_target = center_target, rescale_target = rescale_target,loss_imbalance = loss_imbalance,
        random_state  = random_state, verbose  = verbose) 
        
    def predict(self, X):
        if self.multi_class:
            return self.label_encoder.inverse_transform(np.argmax(self._predict_hidden(X),axis=-1))
        else:
            return self._predict_hidden(X) >= 0.
        
    def decision_function(self, X):
        return self._predict_hidden(X)
    
    def predict_proba(self, X):
        output = self._predict_hidden(X)
        if self.multi_class: 
            probas = softmax_func(output, axis=-1)
            return probas
        else:
            proba = logistic_func(output)
            return np.c_[1.-proba, proba]