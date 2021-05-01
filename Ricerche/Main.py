
from Ricerca import *

f=True
while(f):
    p=True
    print("\nFG:Foggia, BA: Bari, BT: Barletta-Andria-Trani\nLE:Lecce, TA:Taranto, BR:Brindisi")
    while(p):
        print("Inserisci provincia(Sigla):")
        provincia = input()
        if(provincia=="FG"):
            p=grafo.provinciaFG.copy()
            print("I comuni da cui puoi partire sono:",p.keys())
            a = grafo.ospedaliFG
            p=False
        elif(provincia=='BA'):
            p = grafo.provinciaBA.copy()
            print("I comuni da cui puoi partire sono:",p.keys())
            a = grafo.ospedaliBA
            p = False
        elif (provincia == 'TA'):
            p = grafo.provinciaTA.copy()
            print("I comuni da cui puoi partire sono:",p.keys())
            a = grafo.ospedaliTA
            p = False
        elif (provincia == 'LE'):
            p = grafo.provinciaLE.copy()
            print("I comuni da cui puoi partire sono:",p.keys())
            a = grafo.ospedaliLE
            p = False
        elif (provincia == 'BR'):
            p = grafo.provinciaBR.copy()
            print("I comuni da cui puoi partire sono:",p.keys())
            a=grafo.ospedaliBR
            p = False
        elif (provincia == 'BT'):
            p = grafo.provinciaBT.copy()
            print("I comuni da cui puoi partire sono:",p.keys())
            a=grafo.ospedaliBT
            p = False
        else: print("Errore! Riprova")

    print("Inserisci comune di partenza:")
    stato_iniziale=input()
    print('Gli ospedali di provincia sono:',a)
    print("Inserisci comune di arrivo:")
    stato_finale=input()

    print("Quale algorimto vuoi usare?:\n1)DLS\n2)UCS\n3)DFS\n4)BFS")
    s=input()
    if(s=="1"):
        Y= RicercaDLS(stato_iniziale,stato_finale,50,provincia)
        irs=Y.run()
        print(irs)
    elif(s=="2"):
        Y=RicercaUCS(stato_iniziale,stato_finale,provincia)
        irs = Y.run()
        print(irs)
    elif(s=="3"):
        Y=RicercaDFS(stato_iniziale,stato_finale,provincia)
        irs = Y.run()
        print(irs)
    elif(s=="4"):
        Y=RicercaBFS(stato_iniziale,stato_finale,provincia)
        irs = Y.run()
        print(irs)
    else: print("Errore")

    print("Vuoi Ricominciare?(si/no):")
    o=input()
    if(o=="si"):
        f=True
    else:
        f=False






