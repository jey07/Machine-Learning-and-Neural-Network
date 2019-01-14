#h_w- hidden layer weights
#bs- bias for internal layer
#out_bs - output bias

forwardpropogate <- function(inp,weights,bias) {
    
hid_act <- tanh(inp*weights[,1] + bias[,1]) 
out_act <- sum(hid_act*weights[,2] + bias[,2])
return(list(hid_act,out_act))
}

backwardpropogate <- function(differential_cost,activatn_inner,
                              activatn_outr,weights,bias,inp) {

#change in weight from multi-node hidden layer to single output layer
    changact_wrt_inpt_lastl <- c()
    changcost_wrt_actprev <- c()
    
       num <- (exp(activatn_outr) - exp(-(activatn_outr)))^2
       denom <- (exp(activatn_outr) + exp(-(activatn_outr)))^2
      
       changact_wrt_inpt_lastl <- 1- (num/denom)
    
    changcost_wrt_actprev  <- weights[,2]*differential_cost
    
    weights[,2] <<- weights[,2] - 0.004*(activatn_inner)*differential_cost
    bias [,2]  <<- bias[,2] - 0.004* differential_cost
    
    #change in weight from single node input to multi node hidden layer
     changact_wrt_inpt_hiddenl <- c()
    
        for(j in 1:length(activatn_inner)) {
            num <- (exp(activatn_inner[j]) - exp(-(activatn_inner[j])))^2
            denom <- (exp(activatn_inner[j]) + exp(-(activatn_inner[j])))^2
            
            changact_wrt_inpt_hiddenl[j] <- 1- (num/denom)
        }
    
    weights[,1] <<- weights[,1] - 0.005*changact_wrt_inpt_hiddenl*inp*changcost_wrt_actprev
    bias[,1] <<- bias[,1] - 0.005*changact_wrt_inpt_hiddenl*changcost_wrt_actprev
}


set.seed(1234567890)
Var <- runif(200, 0, 8)
trva <- data.frame(Var, Sin=sin(Var))
tr <- trva[1:150,] # Training
va <- trva[151:200,] # Validation


#Randomize hidden layer weights
h_w <- runif(10,-1,1)
#Randomize output layer weights
o_w <- runif(10,-1,1)
#Weight matrix
weights <- cbind(hidden_weight=h_w,output_weight=o_w)


inp_bs <- runif(10,-1,1) #Randomize input bias
out_bs <- runif(10,-1,1) #Randomize output layer bias

bias <- cbind(input_bias=inp_bs,output_bias =out_bs)  #bias matrix


cum_cost <- c()
#compare <- matrix(ncol=,nrow=25)
for (k in 1:5000){      #Number of iterations
    cost <- c()
    y<-c()
    for(i in 1:150) {
        output <- forwardpropogate(tr[i,1],weights,bias)
        activatn_inner <- output[[1]]
        activatn_outr <- output[[2]]
        
        y[i] <- activatn_outr
        cost[i] <- 1/2*((activatn_outr-tr[i,2])^2)
        
        differential_cost <- (activatn_outr-tr[i,2])
        backwardpropogate(differential_cost,activatn_inner,
                           activatn_outr,weights,bias,tr[i,1])
    }
    
    #compare[,k] <- y
    cum_cost[k] <- sum(cost)
    
}

plot(x=1:5000,y=cum_cost[1:5000])
{plot(x=tr[[1]],y=y,ylim= c(-1,1))
points(x=tr[[1]],y=tr[[2]],col="red")}

z<-c()
#Validating data  
for(i in 1:50) {
    output <- forwardpropogate(va[i,1],weights,bias)
    activatn_outr <- output[[2]]
    z[i] <- activatn_outr
}

{plot(x=va[[1]],y=z,ylim= c(-1,1))
    points(x=va[[1]],y=va[[2]],col="red")}
