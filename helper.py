import matplotlib.pyplot as plt
from IPython import display


COLOR_A = ('#BBA0CA')
COLOR_B = ('#99F7AB')

plt.ion()

def plot(scoresA, mean_scoresA, scoresB, mean_scoresB):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Points')
    plt.plot(scoresA , color = (COLOR_A))   
    plt.plot(mean_scoresA, linestyle='dashed', color = (COLOR_A))
    plt.plot(scoresB,  color = (COLOR_B))
    plt.plot(mean_scoresB,linestyle='dashed', color = (COLOR_B))
    plt.ylim(ymin=0)
    plt.text(len(scoresA)-1, scoresA[-1], str(scoresA[-1]))
    plt.text(len(scoresB)-1, scoresB[-1], str(scoresB[-1]))
    plt.text(len(mean_scoresA)-1, mean_scoresA[-1], str(mean_scoresA[-1]))
    plt.text(len(mean_scoresB)-1, mean_scoresB[-1], str(mean_scoresB[-1]))
    plt.show(block=False)
    plt.pause(.1)