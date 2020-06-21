import matplotlib.pyplot as plt 



def plot_data(path,data,label,games_number):
    print('Graph HAAAAAAAAAAAAAASS BEEEEEENNN DRAAAWWWWNNNN')
    colors ={
        'Q':'#33BEFF',
        'R':'#139C3E',
        'T':'#9C133A',
        'C':'#7CB1C5'
    }
    plt.plot(data,label=f'label of {games_number} games',color=colors[label[0]])
    plt.xlabel("Number of games")
    plt.ylabel(label)
    plt.savefig(path)
    plt.close()
