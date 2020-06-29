import matplotlib.pyplot as plt 



def plot_data(path,data,label,games_number):
    print('Graph HAAAAAAAAAAAAAASS BEEEEEENNN DRAAAWWWWNNNN')
    colors ={
        'Q':'#33BEFF',
        'R':'#139C3E',
        'T':'#9C133A',
        'C':'#7CB1C5'
    }
    plt.plot(data,label=f'label of {games_number} games',color=colors.get(label[0],'black'))
    plt.xlabel("Number of games")
    plt.ylabel(label)
    plt.savefig(path)
    plt.close()



def hist_data(path,data,label,bins=None):
    if bins ==None:
        bins= list(range(1,len(data)+1))
    plt.hist(x=data,bins=bins,rwidth=0.5)
    plt.xlabel(label)
    plt.ylabel(f"Frequency of {label}")
    plt.savefig(path)
    plt.close()