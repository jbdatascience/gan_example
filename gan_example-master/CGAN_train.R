#####################################################
### Training module for GAN (CGAN_train.R)
#####################################################

devices<- mx.cpu()

data_shape_G<- c(1, 1, 10, batch_size)
data_shape_D<- c(28, 28, 1, batch_size)
digit_shape_D<- c(10, batch_size)

mx.metric.binacc <- mx.metric.custom("binacc", function(label, pred) {
  res <- mean(label==round(pred))
  return(res)
})

mx.metric.logloss <- mx.metric.custom("logloss", function(label, pred) {
  res <- mean(label*log(pred)+(1-label)*log(1-pred))
  return(res)
})

##############################################
### Define iterators
iter_G<- G_iterator(batch_size = batch_size)
iter_D<- D_iterator(batch_size = batch_size)

exec_G<- mx.simple.bind(symbol = G_sym, data=data_shape_G, ctx = devices, grad.req = "write")
exec_D<- mx.simple.bind(symbol = D_sym, data=data_shape_D, digit=digit_shape_D, ctx = devices, grad.req = "write")

### initialize parameters - To Do - personalise each layer
initializer<- mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 3)

arg_param_ini_G<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(G_sym, data=data_shape_G)$arg.shapes, ctx = mx.cpu())
aux_param_ini_G<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(G_sym, data=data_shape_G)$aux.shapes, ctx = mx.cpu())

arg_param_ini_D<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(D_sym, data=data_shape_D, digit=digit_shape_D)$arg.shapes, ctx = mx.cpu())
aux_param_ini_D<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(D_sym, data=data_shape_D, digit=digit_shape_D)$aux.shapes, ctx = mx.cpu())

mx.exec.update.arg.arrays(exec_G, arg_param_ini_G, match.name=TRUE)
mx.exec.update.aux.arrays(exec_G, aux_param_ini_G, match.name=TRUE)

mx.exec.update.arg.arrays(exec_D, arg_param_ini_D, match.name=TRUE)
mx.exec.update.aux.arrays(exec_D, aux_param_ini_D, match.name=TRUE)

input_names_G <- mxnet:::mx.model.check.arguments(G_sym)
input_names_D <- mxnet:::mx.model.check.arguments(D_sym)


###################################################
#initialize optimizers
optimizer_G<-mx.opt.create(name = "adadelta",
                           rho=0.92, 
                           epsilon = 1e-6, 
                           wd=0, 
                           rescale.grad=1/batch_size, 
                           clip_gradient=1)

updater_G<- mx.opt.get.updater(optimizer = optimizer_G, weights = exec_G$ref.arg.arrays)

optimizer_D<-mx.opt.create(name = "adadelta",
                           rho=0.92, 
                           epsilon = 1e-6, 
                           wd=0, 
                           rescale.grad=1/batch_size, 
                           clip_gradient=1)
updater_D<- mx.opt.get.updater(optimizer = optimizer_D, weights = exec_D$ref.arg.arrays)

####################################
#initialize metric
metric_G<- mx.metric.binacc
metric_G_value<- metric_G$init()

metric_D<- mx.metric.binacc
metric_D_value<- metric_D$init()

iteration<- 1
iter_G$reset()
iter_D$reset()


for (iteration in 1:2400) {
  
  iter_G$iter.next()
  iter_D$iter.next()
  
  ### Random input to Generator to produce fake sample
  G_values <- iter_G$value()
  G_data <- G_values[input_names_G]
  mx.exec.update.arg.arrays(exec_G, arg.arrays = G_data, match.name=TRUE)
  mx.exec.forward(exec_G, is.train=T)
  
  ### Feed Discriminator with Concatenated Generator images and real images
  ### Random input to Generator
  D_data_fake <- exec_G$ref.outputs$G_sym_output
  D_digit_fake <- G_values$data %>% mx.nd.Reshape(shape=c(-1, batch_size))
  
  D_values <- iter_D$value()
  D_data_real <- D_values$data
  D_digit_real <- D_values$digit
  
  ### Train loop on fake
  mx.exec.update.arg.arrays(exec_D, arg.arrays = list(data=D_data_fake, digit=D_digit_fake, label=mx.nd.array(rep(0, batch_size))), match.name=TRUE)
  mx.exec.forward(exec_D, is.train=T)
  mx.exec.backward(exec_D)
  update_args_D<- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_D, update_args_D, skip.null=TRUE)
  
  metric_D_value <- metric_D$update(label = mx.nd.array(rep(0, batch_size)), exec_D$ref.outputs[["D_sym_output"]], metric_D_value)
  
  ### Train loop on real
  mx.exec.update.arg.arrays(exec_D, arg.arrays = list(data=D_data_real, digit=D_digit_real, label=mx.nd.array(rep(1, batch_size))), match.name=TRUE)
  mx.exec.forward(exec_D, is.train=T)
  mx.exec.backward(exec_D)
  update_args_D<- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_D, update_args_D, skip.null=TRUE)
  
  metric_D_value <- metric_D$update(mx.nd.array(rep(1, batch_size)), exec_D$ref.outputs[["D_sym_output"]], metric_D_value)
  
  ### Update Generator weights - use a seperate executor for writing data gradients
  exec_D_back<- mxnet:::mx.symbol.bind(symbol = D_sym, arg.arrays = exec_D$arg.arrays, aux.arrays = exec_D$aux.arrays, grad.reqs = rep("write", length(exec_D$arg.arrays)), ctx = devices)
  mx.exec.update.arg.arrays(exec_D_back, arg.arrays = list(data=D_data_fake, digit=D_digit_fake, label=mx.nd.array(rep(1, batch_size))), match.name=TRUE)
  mx.exec.forward(exec_D_back, is.train=T)
  mx.exec.backward(exec_D_back)
  D_grads<- exec_D_back$ref.grad.arrays$data
  mx.exec.backward(exec_G, out_grads=D_grads)
  
  update_args_G<- updater_G(weight = exec_G$ref.arg.arrays, grad = exec_G$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_G, update_args_G, skip.null=TRUE)
  
  ### Update metrics
  #metric_G_value <- metric_G$update(values[[label_name]], exec_G$ref.outputs[[output_name]], metric_G_value)
  
  if (iteration %% 25==0){
    D_metric_result <- metric_D$get(metric_D_value)
    cat(paste0("[", iteration, "] ", D_metric_result$name, ": ", D_metric_result$value, "\n"))
  }
  
  if (iteration==1 | iteration %% 100==0){
    
    metric_D_value<- metric_D$init()
    
    par(mfrow=c(3,3), mar=c(0.1,0.1,0.1,0.1))
    for (i in 1:9) {
      img <- as.array(exec_G$ref.outputs$G_sym_output)[,,,i]
      plot(as.cimg(img), axes=F)
    }

    print(as.numeric(as.array(G_values$digit)))
    print(as.numeric(as.array(D_values$label)))
    
  }
}

mx.symbol.save(D_sym, filename = "models/D_sym_model_v1.json")
mx.nd.save(exec_D$arg.arrays, filename = "models/D_aux_params_v1.params")
mx.nd.save(exec_D$aux.arrays, filename = "models/D_aux_params_v1.params")

mx.symbol.save(G_sym, filename = "models/G_sym_model_v1.json")
mx.nd.save(exec_G$arg.arrays, filename = "models/G_arg_params_v1.params")
mx.nd.save(exec_G$aux.arrays, filename = "models/G_aux_params_v1.params")


### Inference
G_sym<- mx.symbol.load("models/G_sym_model_v1.json")
G_arg_params<- mx.nd.load("models/G_arg_params_v1.params")
G_aux_params<- mx.nd.load("models/G_aux_params_v1.params")

digit<- mx.nd.array(rep(9, times=batch_size))
data<- mx.nd.one.hot(indices = digit, depth = 10)
data<- mx.nd.reshape(data = data, shape = c(1,1,-1, batch_size))

exec_G<- mx.simple.bind(symbol = G_sym, data=data_shape_G, ctx = devices, grad.req = "null")
mx.exec.update.arg.arrays(exec_G, G_arg_params, match.name=TRUE)
mx.exec.update.arg.arrays(exec_G, list(data=data), match.name=TRUE)
mx.exec.update.aux.arrays(exec_G, G_aux_params, match.name=TRUE)

mx.exec.forward(exec_G, is.train=F)

par(mfrow=c(3,3), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:9) {
  img <- as.array(exec_G$ref.outputs$G_sym_output)[,,,i]
  plot(as.cimg(img), axes=F)
}

#-----------------------------------------------------------------------------------------
#
#[1] 0 4 5 3 2 2 5 4 9 2 7 8 7 8 4 0 5 7 8 0 6 6 0 4 3 2 8 2 8 3 8 3 4 8 1 9 1 9 1 3 1 8 6 2 4 3 9
#[48] 2 4 1 4 8 5 2 2 3 2 7 1 5 4 4 0 3
#[1] 4 4 6 3 3 8 9 6 4 7 7 8 6 3 6 0 4 9 9 4 7 5 7 2 4 7 1 3 1 0 2 2 5 0 5 5 3 1 8 4 5 2 3 7 2 8 8
#[48] 5 2 2 0 8 1 9 2 5 4 7 2 8 9 6 2 7
#[25] binacc: 0.914388020833333
#[50] binacc: 0.957589285714286
#[75] binacc: 0.971706081081081
#[100] binacc: 0.978614267676768
#[1] 8 3 1 5 4 8 6 2 7 8 5 4 7 1 1 2 3 6 7 8 9 8 1 2 3 8 7 5 2 5 9 2 3 8 6 3 0 2 0 6 3 1 4 9 8 1 5
#[48] 3 1 8 8 7 0 9 5 0 9 2 0 4 4 4 3 3
#[1] 7 3 9 9 0 1 9 9 9 8 9 6 0 3 1 8 3 4 4 5 6 3 1 8 8 8 8 5 3 9 7 7 3 4 2 1 9 9 1 5 9 7 3 3 7 3 1
#[48] 1 0 7 1 8 9 5 4 6 9 5 5 9 6 9 7 3
#[125] binacc: 0.6353125
#[150] binacc: 0.7196875
#[175] binacc: 0.726041666666667
#[200] binacc: 0.69609375
#[1] 2 9 8 2 7 5 5 9 8 5 7 6 5 8 4 5 8 2 6 5 4 0 6 6 5 0 7 4 9 0 0 5 1 1 4 7 4 7 4 0 1 0 5 0 4 1 5
#[48] 2 4 5 9 7 2 7 0 4 1 7 4 2 2 9 7 7
#[1] 4 7 4 9 8 7 5 4 7 2 1 9 0 7 6 4 1 6 3 0 1 8 6 3 7 9 3 5 2 8 5 5 9 4 4 8 5 2 3 8 1 5 4 3 9 7 9
#[48] 8 3 8 0 7 3 9 0 9 0 8 9 7 7 1 5 7
#[225] binacc: 0.531875
#[250] binacc: 0.52515625
#[275] binacc: 0.540416666666667
#[300] binacc: 0.5875
#[1] 1 5 9 7 2 5 6 8 1 9 2 1 4 1 8 6 1 5 7 5 5 0 1 3 5 2 4 7 9 2 5 8 6 6 4 5 9 1 1 1 6 4 5 9 4 1 1
#[48] 5 8 9 1 8 8 2 6 4 4 2 1 3 4 2 6 3
#[1] 7 0 7 3 6 3 9 9 3 9 2 3 1 2 1 1 6 7 3 7 2 7 8 5 7 0 7 3 0 7 1 0 8 8 5 4 4 7 2 7 4 3 0 7 9 2 4
#[48] 2 3 1 7 1 8 1 2 9 5 5 3 6 3 8 5 8
#[325] binacc: 0.854375
#[350] binacc: 0.86328125
#...

#-----------------------------------------------------------------------------------------
