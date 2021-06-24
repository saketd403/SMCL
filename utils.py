

def on_device(inputs,labels,num_models):

    inputs_ls=[]
    labels_ls=[]

    for _ in range(num_models):

        inputs_ls.append(inputs.clone().cuda())
        labels_ls.append(labels.clone().cuda())

    return (inputs_ls,labels_ls)


