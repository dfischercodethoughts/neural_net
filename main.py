import neural_net

if __name__ == "__main__":
    ################ PART 2.c
    net = neural_net.Neural_Network(10, 1, "mnist_all.mat", 784, .001)
    start = net
    net.batch_size = 1000
    num_iterations = 1000
    print("beginning training")
    for i in range(num_iterations):
        print("beginning iteration " + str(i) + "...")
        net.perform_iteration(debug=False)
        print("ending iteration.")
    print("done. running test on train data...")
    train_acc = net.run_test_on_train_data()
    print("accuracy: " + str(train_acc))
    print("done.")
    print("running test on test data...")
    test_acc = net.run_test_on_test_data()
    print("accuracy: " + str(test_acc))
    print("done.")

    #########################Part 2.di
    # net1 = Neural_Network(10,1,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    # net2 = Neural_Network(10,2,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    # net3 = Neural_Network(10,3,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    # net4 = Neural_Network(10,4,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    # net5 = Neural_Network(10,5,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    # net6 = Neural_Network(10,6,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    # net7 = Neural_Network(10,7,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    # net8 = Neural_Network(10,8,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    # net9 = Neural_Network(10,9,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    # net10 = Neural_Network(10,10,"COMPSCI_671_Homework_4\mnist_all.mat",784,.1)
    #
    # net1.perform_iteration()
    # net2.perform_iteration()
    # net3.perform_iteration()
    # net4.perform_iteration()
    # net5.perform_iteration()
    # net6.perform_iteration()
    # net7.perform_iteration()
    # net8.perform_iteration()
    # net9.perform_iteration()
    # net10.perform_iteration()
    #
    # #toplt1 = max(np.array(net1.first_weight_gradient).flatten())
    # plt.yscale('log')
    # plt.title("Max Layer One Gradient")
    # plt.ylabel("Max Gradient")
    # plt.xlabel("Number of Layers")
    # plt.plot(1,max(np.array(net1.first_weight_gradient).flatten()),'b.')
    # plt.plot(2,max(np.array(net2.first_weight_gradient).flatten()),'b.')
    # plt.plot(3,max(np.array(net3.first_weight_gradient).flatten()),'b.')
    # plt.plot(4,max(np.array(net4.first_weight_gradient).flatten()),'b.')
    # plt.plot(5,max(np.array(net5.first_weight_gradient).flatten()),'b.')
    # plt.plot(6,max(np.array(net6.first_weight_gradient).flatten()),'b.')
    # plt.plot(7,max(np.array(net7.first_weight_gradient).flatten()),'b.')
    # plt.plot(8,max(np.array(net8.first_weight_gradient).flatten()),'b.')
    # plt.plot(9,max(np.array(net9.first_weight_gradient).flatten()),'b.')
    # plt.plot(10,max(np.array(net10.first_weight_gradient).flatten()),'b.')
    # plt.savefig('test')
    # plt.show()
    #

# %%



