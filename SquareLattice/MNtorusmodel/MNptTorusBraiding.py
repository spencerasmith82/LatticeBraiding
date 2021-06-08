#This is a quick set of classes to represent loops on the MxN lattice on a Torus and the action of braid generators on loops


import numpy as np
import copy
import math
from scipy.optimize import curve_fit
import random

#first let's create a loop class
class Loop:
    
    #constructor
    def __init__(self,SZ,loopweights = None):
        self.weightlist = []
        self.SZ = []
        self.SZ.append(SZ[0])
        self.SZ.append(SZ[1])
        if loopweights is None:
            for i in range(self.SZ[0]):
                self.weightlist.append([])
                for j in range(self.SZ[1]):
                    self.weightlist[i].append([])
                    self.weightlist[i][j].append(1)
                    self.weightlist[i][j].append(2)
                    self.weightlist[i][j].append(1)
        else:
            counter = 0
            for i in range(self.SZ[0]):
                self.weightlist.append([])
                for j in range(self.SZ[1]):
                    self.weightlist[i].append([])
                    for k in range(3):
                        self.weightlist[i][j].append(loopweights[counter])
                        counter +=1
            
    def GetShearWeight(self,i,j,k):
        I0 = (self.SZ[0]+i)%self.SZ[0]
        J0 = (self.SZ[1]+j)%self.SZ[1]   
        if k == 0:
            Im1 = (self.SZ[0]+i-1)%self.SZ[0]
            Jp1 = (self.SZ[1]+j+1)%self.SZ[1]
            return (-self.weightlist[Im1][J0][2] + self.weightlist[Im1][J0][1] - self.weightlist[I0][Jp1][2] + self.weightlist[I0][J0][1])
        elif k == 1:
            Ip1 = (self.SZ[0]+i+1)%self.SZ[0]
            Jp1 = (self.SZ[1]+j+1)%self.SZ[1]
            return (-self.weightlist[I0][J0][0] + self.weightlist[I0][Jp1][2] - self.weightlist[Ip1][J0][0] + self.weightlist[I0][J0][2])
        else:
            Ip1 = (self.SZ[0]+i+1)%self.SZ[0]
            Jm1 = (self.SZ[1]+j-1)%self.SZ[1]
            return (-self.weightlist[I0][J0][1] + self.weightlist[Ip1][J0][0] - self.weightlist[I0][Jm1][1] + self.weightlist[I0][Jm1][0])
        
        
    def WeightTotal(self):
        WeightTot = 0
        for i in range(self.SZ[0]):
            for j in range(self.SZ[1]):
                for k in range(3):
                    WeightTot += abs(self.GetShearWeight(i,j,k))
        return WeightTot
    
    
    
#Each generator will be an object, which can act on a loop
#orientation is either Vertical (default), or Horizontal.  If vertical then the i,j and i+1,j points are switched, if horizontal, then the i,j and i,j+1 points are switched.
#sw tells us the direction of the switch, CCW or counter-clock-wise is default (positive generator), and CW (other) is clockwise
class BraidGenerator:
    
    def __init__(self,point,orientation = "Vertical", sw = "CCW"):
        self.pt = point #the [i,j] indices
        self.orient = orientation
        self.switch = sw
    
    
    def Action(self,LoopIn):
        p1 = Patch(self.orient)
        p1.FillPatch(self.pt[0],self.pt[1],LoopIn)
        if self.orient == "Vertical":
            if self.switch == "CCW":
                p1.CCWswitch()
            else:
                p1.Xmirror()
                p1.CCWswitch()
                p1.Xmirror()
        elif self.orient == "Diagonal":
            if self.switch == "CCW":
                p1 = p1.VDswitch()
                p1.CCWswitch()
                p1 = p1.VDswitch()
            else:
                p1 = p1.VDswitch()
                p1.Xmirror()
                p1.CCWswitch()
                p1.Xmirror()
                p1 = p1.VDswitch()
        else:
            if self.switch == "CCW":
                p1 = p1.VHswitch()
                p1.CCWswitch()
                p1 = p1.VHswitch()
            else:
                p1 = p1.VHswitch()
                p1.Xmirror()
                p1.CCWswitch()
                p1.Xmirror()
                p1 = p1.VHswitch()
        p1.PastePatch(self.pt[0],self.pt[1],LoopIn)

#Total braid action on a loop
def TotalBraidAction(Bin,LoopIn):
    for i in range(len(Bin)):
        for j in range(len(Bin[i])):
            Bin[i][j].Action(LoopIn)
    
        
#this is the topological entropy of a braid
#here the braid is composed of operators, and each operator is composed of a certain number of mutually compatible braid generators.  This top. entropy is scaled by the number of operators
def GetTE(Bin,SZ, tolerance = 1e-10, numitermax = 100,Verbose = False):
    Loop1 = Loop(SZ)
    NumGen = len(Bin)
    numitermin = 6
    LogWeightPrev = math.log(Loop1.WeightTotal())
    for i in range(numitermin):
        TotalBraidAction(Bin,Loop1)
    LogWeight = math.log(Loop1.WeightTotal())
    
    
    iternum = numitermin
    TE = (LogWeight - LogWeightPrev)/NumGen
    TEprev = 0
    
    while np.abs(TE - TEprev) > tolerance and iternum < numitermax:
        iternum += 1
        TotalBraidAction(Bin,Loop1)
        LogWeightPrev = LogWeight
        TEprev = TE
        LogWeight = math.log(Loop1.WeightTotal())
        TE = (LogWeight - LogWeightPrev)/NumGen

    if Verbose:
        if iternum == numitermax:
            print("Braiding Entropy of ", TE, " with tolerance of ", np.abs(TE - TEprev), " after the maximum of ", iternum, " iterations")
        else:
            print("Braiding Entropy of ", TE, " with tolerance of ", np.abs(TE - TEprev), " after ", iternum, " iterations")
    return [TE, np.abs(TE - TEprev)]        
        
    
    
#This creates the subset of intersection coordinates from the loop that can be acted on by the generator
class Patch:
    
    def __init__(self,orientation = "Vertical"):
        self.Mweights = []
        self.weights3 = [0,0,0]
        self.weights2 = [0,0]
        self.orient = orientation
        if self.orient == "Vertical":
            for i in range(3):
                self.Mweights.append([])
                for j in range(2):
                    self.Mweights[i].append([])
                    for k in range(3):
                        self.Mweights[i][j].append([0])
        else:
            for i in range(2):
                self.Mweights.append([])
                for j in range(3):
                    self.Mweights[i].append([])
                    for k in range(3):
                        self.Mweights[i][j].append([0])

    #This fills in the contents of this patch with the intersection coordinates from LoopIn
    def FillPatch(self,iin,jin, LoopIn):
        M = LoopIn.SZ[0]
        N = LoopIn.SZ[1]
        if self.orient == "Vertical":
            for s in range(3):
                self.weights3[s] = LoopIn.weightlist[(M+s+iin-1)%M][(N+jin+1)%N][2]
            for s in range(2):
                self.weights2[s] = LoopIn.weightlist[(M+iin+2)%M][(N+s+jin-1)%N][0]    
            for i in range(3):
                for j in range(2):
                    for k in range(3): 
                        self.Mweights[i][j][k] = LoopIn.weightlist[(M+i+iin-1)%M][(N+j+jin-1)%N][k]
        elif self.orient == "Diagonal":
            for s in range(3):
                self.weights3[s] = LoopIn.weightlist[(M+iin+s)%M][(N+s+jin-1)%N][1]
            for s in range(2):
                self.weights2[s] = LoopIn.weightlist[(M+s+iin+1)%M][(N+jin+2)%N][2]
            for j in range(3):
                for k in range(3): 
                    self.Mweights[1][j][k] = LoopIn.weightlist[(M+iin-1+j)%M][(N+j+jin-1)%N][k]
                    if k == 0:  #this encoding is mixed around a bit, but works as a convention
                        self.Mweights[0][j][k] = LoopIn.weightlist[(M+iin+j)%M][(N+j+jin-1)%N][k]
                    else:
                        self.Mweights[0][j][k] = LoopIn.weightlist[(M+iin+j-2)%M][(N+j+jin-1)%N][k]
        else:   #the Horizontal case
            for s in range(3):
                self.weights3[s] = LoopIn.weightlist[(M+iin+1)%M][(N+s+jin-1)%N][0]
            for s in range(2):
                self.weights2[s] = LoopIn.weightlist[(M+s+iin-1)%M][(N+jin+2)%N][2]
            for i in range(2):
                for j in range(3):
                    for k in range(3): 
                        self.Mweights[i][j][k] = LoopIn.weightlist[(M+i+iin-1)%M][(N+j+jin-1)%N][k]  
    
    #This fills in the given loop with this patch
    def PastePatch(self,iin,jin,Loop):
        M = Loop.SZ[0]
        N = Loop.SZ[1]
        if self.orient == "Vertical":
            for s in range(3):
                Loop.weightlist[(M+s+iin-1)%M][(N+jin+1)%N][2] = self.weights3[s]
            for s in range(2):
                Loop.weightlist[(M+iin+2)%M][(N+s+jin-1)%N][0] = self.weights2[s]
            for i in range(3):
                for j in range(2):
                    for k in range(3): 
                        Loop.weightlist[(M+i+iin-1)%M][(N+j+jin-1)%N][k] = self.Mweights[i][j][k]
        elif self.orient == "Diagonal":
            for s in range(3):
                Loop.weightlist[(M+iin+s)%M][(N+s+jin-1)%N][1] = self.weights3[s]
            for s in range(2):
                Loop.weightlist[(M+s+iin+1)%M][(N+jin+2)%N][2] = self.weights2[s]
            for j in range(3):
                for k in range(3): 
                    Loop.weightlist[(M+iin-1+j)%M][(N+j+jin-1)%N][k] = self.Mweights[1][j][k]
                    if k == 0:  #this encoding is mixed around a bit, but works as a convention
                        Loop.weightlist[(M+iin+j)%M][(N+j+jin-1)%N][k] = self.Mweights[0][j][k]
                    else:
                        Loop.weightlist[(M+iin+j-2)%M][(N+j+jin-1)%N][k] = self.Mweights[0][j][k]
        else:
            for s in range(3):
                Loop.weightlist[(M+iin+1)%M][(N+s+jin-1)%N][0] = self.weights3[s]
            for s in range(2):
                Loop.weightlist[(M+s+iin-1)%M][(N+jin+2)%N][2] = self.weights2[s]
            for i in range(2):
                for j in range(3):
                    for k in range(3): 
                        Loop.weightlist[(M+i+iin-1)%M][(N+j+jin-1)%N][k] = self.Mweights[i][j][k]
    
    #this takes the patch and returns a new patch that is rotated to vertical if horizontal, or vice versa
    def VHswitch(self):
        if self.orient == "Vertical":
            Patchout = Patch(orientation = "Horizontal")
            Patchout.weights2[0] = self.weights2[1]
            Patchout.weights2[1] = self.weights2[0]
            for i in range(3):
                Patchout.weights3[i] = self.Mweights[i][0][2]
                Patchout.Mweights[1][i][2] = self.Mweights[i][0][0]
                Patchout.Mweights[0][i][2] = self.Mweights[i][1][0]
                Patchout.Mweights[1][i][0] = self.Mweights[i][1][2]
                Patchout.Mweights[0][i][0] = self.weights3[i]
            for i in range(2):
                Patchout.Mweights[1][i][1] = WHmove([self.Mweights[i][0][0],self.Mweights[i][1][2],self.Mweights[i+1][0][0],self.Mweights[i][0][2],self.Mweights[i][0][1]])
                Patchout.Mweights[0][i][1] = WHmove([self.Mweights[i][1][0],self.weights3[i],self.Mweights[i+1][1][0],self.Mweights[i][1][2],self.Mweights[i][1][1]])
            Patchout.Mweights[1][2][1] = WHmove([self.Mweights[2][0][0],self.Mweights[2][1][2],self.weights2[0],self.Mweights[2][0][2],self.Mweights[2][0][1]])
            Patchout.Mweights[0][2][1] = WHmove([self.Mweights[2][1][0], self.weights3[2], self.weights2[1],self.Mweights[2][1][2],self.Mweights[2][1][1]])
            
            return Patchout
            
        elif self.orient == "Horizontal":
            Patchout = Patch()
            Patchout.weights2[0] = self.weights2[1]
            Patchout.weights2[1] = self.weights2[0]
            for i in range(3):
                Patchout.weights3[i] = self.Mweights[0][i][0]
                Patchout.Mweights[i][0][0] = self.Mweights[1][i][2]
                Patchout.Mweights[i][1][0] = self.Mweights[0][i][2]
                Patchout.Mweights[i][1][2] = self.Mweights[1][i][0]
                Patchout.Mweights[i][0][2] = self.weights3[i]
            for i in range(2):
                Patchout.Mweights[i][0][1] = WHmove([self.weights3[i],self.Mweights[1][i][2],self.Mweights[1][i][0],self.Mweights[1][i+1][2],self.Mweights[1][i][1]])
                Patchout.Mweights[i][1][1] = WHmove([self.Mweights[1][i][0],self.Mweights[0][i][2],self.Mweights[0][i][0],self.Mweights[0][i+1][2],self.Mweights[0][i][1]])
            Patchout.Mweights[2][0][1] = WHmove([self.weights3[2],self.Mweights[1][2][2],self.Mweights[1][2][0],self.weights2[1],self.Mweights[1][2][1]])
            Patchout.Mweights[2][1][1] = WHmove([self.Mweights[1][2][0],self.Mweights[0][2][2],self.Mweights[0][2][0],self.weights2[0],self.Mweights[0][2][1]])
            
            return Patchout
        else:
            print("Can't use VHswitch with a Diagonal Patch")
 

    #this takes the patch and returns a new patch that is rotated to vertical if Diagonal, or vice versa
    def VDswitch(self):
        if self.orient == "Vertical":
            Patchout = Patch(orientation = "Diagonal")
            Patchout.weights2[0] = self.weights2[1]
            Patchout.weights2[1] = self.weights2[0]
            for s in range(3):
                Patchout.weights3[s] = self.Mweights[s][0][2]
                Patchout.Mweights[0][s][1] = self.weights3[s]
                Patchout.Mweights[0][s][0] = self.Mweights[s][0][1]
                Patchout.Mweights[0][s][2] = self.Mweights[s][1][0]
                Patchout.Mweights[1][s][0] = self.Mweights[s][1][1]
                Patchout.Mweights[1][s][1] = self.Mweights[s][1][2]
                Patchout.Mweights[1][s][2] = self.Mweights[s][0][0]
            return Patchout
        elif self.orient == "Diagonal":
            Patchout = Patch(orientation = "Vertical")
            Patchout.weights2[0] = self.weights2[1]
            Patchout.weights2[1] = self.weights2[0]
            for s in range(3):
                Patchout.Mweights[s][0][2] = self.weights3[s]
                Patchout.weights3[s] = self.Mweights[0][s][1]
                Patchout.Mweights[s][0][1] = self.Mweights[0][s][0]
                Patchout.Mweights[s][1][0] = self.Mweights[0][s][2]
                Patchout.Mweights[s][1][1] = self.Mweights[1][s][0]
                Patchout.Mweights[s][1][2] = self.Mweights[1][s][1]
                Patchout.Mweights[s][0][0] = self.Mweights[1][s][2]
            return Patchout
        else:
            print("Can't use VDswitch with a Horizontal Patch")

    
    #This makes a copy of this patch and returns it
    def PatchCopy(self):
        PatchOut = Patch(orientation = self.orient)
        PatchOut.weights2 = [x for x in self.weights2]
        PatchOut.weights3 = [x for x in self.weights3]
        for i in range(len(self.Mweights)):
            for j in range(len(self.Mweights[0])):
                for k in range(3):
                    PatchOut.Mweights[i][j][k] = self.Mweights[i][j][k]
        return PatchOut
        
    
    #this takes a vertically oriented patch and flips it along the central vertical axis
    def Xmirror(self):
        if self.orient == "Vertical":
            Pcopy = self.PatchCopy()
            for i in range(3):
                self.Mweights[i][0][0] = Pcopy.Mweights[i][1][0]
                self.Mweights[i][1][0] = Pcopy.Mweights[i][0][0]
                self.Mweights[i][0][2] = Pcopy.weights3[i]
                self.weights3[i] = Pcopy.Mweights[i][0][2]
                
            self.weights2[0] = Pcopy.weights2[1]
            self.weights2[1] = Pcopy.weights2[0]
            #now for the diagonals ... 
            for i in range(2):
                self.Mweights[i][0][1] = WHmove([Pcopy.Mweights[i][1][0],Pcopy.weights3[i],Pcopy.Mweights[i+1][1][0],Pcopy.Mweights[i][1][2],Pcopy.Mweights[i][1][1]])
                self.Mweights[i][1][1] = WHmove([Pcopy.Mweights[i][0][0],Pcopy.Mweights[i][1][2],Pcopy.Mweights[i+1][0][0],Pcopy.Mweights[i][0][2],Pcopy.Mweights[i][0][1]])
            self.Mweights[2][0][1] = WHmove([Pcopy.Mweights[2][1][0],Pcopy.weights3[2],Pcopy.weights2[1],Pcopy.Mweights[2][1][2],Pcopy.Mweights[2][1][1]])
            self.Mweights[2][1][1] = WHmove([Pcopy.Mweights[2][0][0],Pcopy.Mweights[2][1][2],Pcopy.weights2[0],Pcopy.Mweights[2][0][2],Pcopy.Mweights[2][0][1]])    
            
        else:
            print("Patch is horizontally oriented, can't mirror along y-axis")
    
    #this changes the patch (Vertical only) based on the action of a braid generator (ccw switch of the central two points)
    def CCWswitch(self):
        E211p = WHmove([self.Mweights[2][1][0],self.weights3[2],self.weights2[1],self.Mweights[2][1][2],self.Mweights[2][1][1]])
        E001p = WHmove([self.Mweights[0][0][0],self.Mweights[0][1][2],self.Mweights[1][0][0],self.Mweights[0][0][2],self.Mweights[0][0][1]])
        
        E210p = WHmove([self.Mweights[1][1][2],self.Mweights[1][1][1],E211p,self.Mweights[2][1][2],self.Mweights[2][1][0]])
        E100p = WHmove([self.Mweights[1][1][2],self.Mweights[1][0][1],E001p,self.Mweights[0][1][2],self.Mweights[1][0][0]])
        
        E212p = WHmove([self.Mweights[1][1][2],E210p,self.Mweights[2][0][1],self.Mweights[2][0][0],self.Mweights[2][1][2]])
        E012p = WHmove([self.Mweights[1][1][2],E100p,self.Mweights[0][1][1],self.Mweights[1][1][0],self.Mweights[0][1][2]])
        
        E200p = WHmove([self.Mweights[1][1][2],E212p,self.Mweights[1][0][2],self.Mweights[1][0][1],self.Mweights[2][0][0]])
        E110p = WHmove([self.Mweights[1][1][2],E012p,self.weights3[1],self.Mweights[1][1][1],self.Mweights[1][1][0]])
        
        E211pp = WHmove([self.weights3[2],self.weights2[1],E210p,self.Mweights[1][1][1],E211p])
        E001pp = WHmove([self.Mweights[0][0][2],self.Mweights[0][0][0],E100p,self.Mweights[1][0][1],E001p])
        
        self.Mweights[0][0][1] = E001pp
        self.Mweights[0][1][2] = E100p
        self.Mweights[1][0][0] = self.Mweights[1][0][1]
        self.Mweights[1][0][1] = E200p
        self.Mweights[1][1][0] = E012p
        self.Mweights[2][1][0] = self.Mweights[1][1][1]
        self.Mweights[1][1][1] = E110p
        self.Mweights[2][0][0] = E212p
        self.Mweights[2][1][1] = E211pp
        self.Mweights[2][1][2] = E210p
        

            
#general function that returns the new weight of the diagonal after a Whitehead move
#W[0-3] are the perimeter points in ccw order, W[4] is the old diagonal
def WHmove(W):
    return max(W[0]+W[2],W[1]+W[3]) - W[4]
            
            
#we would like to generate a random maximal matching.  This will enable a random maximal collection of mutually compatible braid generators (though this does not give a perfect matching that we will need to get the braid operators).  Still this will enable a type of random braiding analysis.
#this function takes in [m,n] m,n>=4, which defines the size of our square torus lattice. and outputs a random maximal matching
def MaximalMatching(Tsize):
    M = Tsize[0]
    N = Tsize[1]
    #first we create the list, which holds a list of neighbors.  each lattice site is labeled by i*N+j
    Clist = []
    for i in range(M):
        for j in range(N):
            Clist.append([[],[]])
            Clist[-1][0] = 4
            Clist[-1][1].append(i*N+(j+1)%N)  #right
            Clist[-1][1].append(i*N+(N+j-1)%N)  #left
            Clist[-1][1].append(((i+1)%M)*N+j)  #up
            Clist[-1][1].append(((M+i-1)%M)*N+j)  #down
    
    Elist = []   #This is the list that holds the left [i,j] point of an edge if horizontal and the bottom [i,j] point if the edge is vertical.  It also stores "Horizontal" or "Vertical" as the second piece of information
    ListSize = len(Clist)
    while ListSize > 0:
        stpoint = random.randint(0,ListSize-1)  #the random point that we will use to find an edge to add to our collection
        #print("List Size", ListSize)
        #print("start point", stpoint)
        
        actualpoint = 0
        counter = -1
        for i in range(len(Clist)):
            if not Clist[i][0] == 0:
                counter += 1
            if counter >= stpoint:
                actualpoint = i
                break
        
        #now the actual point is actualpoint.  We want to randomly choose the available neighboring point
        #first find the set of available points
        #print("Actual Point",actualpoint)
        #print("Clist",Clist)
        avpoints = []
        for i in range(4):
            if not Clist[actualpoint][1][i] is None:
                avpoints.append(i)
        randavpta = 0
        #print("avpoints",avpoints)
        #print(len(avpoints))
        if len(avpoints) > 1:
            randavpta = random.randint(0,len(avpoints)-1)
        randavptb = avpoints[randavpta]
        randavptc = Clist[actualpoint][1][randavptb]
        #now get the [i,j] coordinates of actualpoint
        jpt = actualpoint%N
        ipt = (actualpoint-jpt)//N
        if randavptb == 0: #right
            Elist.append([[ipt,jpt],"Horizontal"])
        elif randavptb == 1:  #left
            Elist.append([[ipt,(N+jpt-1)%N],"Horizontal"])
        elif randavptb == 2:  #up
            Elist.append([[ipt,jpt],"Vertical"])
        else:  #down
            Elist.append([[(M+ipt-1)%M,jpt],"Vertical"])
        #now we clean up Clist
        DualList = [1,0,3,2]
        for i in range(4):
            if not Clist[actualpoint][1][i] is None:
                Clist[Clist[actualpoint][1][i]][0] -= 1
                Clist[Clist[actualpoint][1][i]][1][DualList[i]] = None
        Clist[actualpoint][0] = 0
        Clist[actualpoint][1] = [None,None,None,None]
        #now do this for randavptc
        for i in range(4):
            if not Clist[randavptc][1][i] is None:
                Clist[Clist[randavptc][1][i]][0] -= 1
                Clist[Clist[randavptc][1][i]][1][DualList[i]] = None
        Clist[randavptc][0] = 0
        Clist[randavptc][1] = [None,None,None,None]
        #now find the new ListSize
        ListSize = 0
        for i in range(len(Clist)):
            if Clist[i][0] > 0:
                ListSize += 1
        #print(Elist)
    return Elist
        

#This creates a maximal matching, just as above, but for the triangular lattice (all six edges from a given point are equally likely)
def MaximalMatchingTri(Tsize):
    M = Tsize[0]
    N = Tsize[1]
    #first we create the list, which holds a list of neighbors.  each lattice site is labeled by i*N+j
    Clist = []
    for i in range(M):
        for j in range(N):
            Clist.append([[],[]])
            Clist[-1][0] = 6
            Clist[-1][1].append(i*N+(j+1)%N)  #right
            Clist[-1][1].append(i*N+(N+j-1)%N)  #left
            Clist[-1][1].append(((i+1)%M)*N+j)  #up
            Clist[-1][1].append(((M+i-1)%M)*N+j)  #down
            Clist[-1][1].append(((i+1)%M)*N+(j+1)%N)  #up-right
            Clist[-1][1].append(((M+i-1)%M)*N+(N+j-1)%N)  #down-left
    
    Elist = []   #This is the list that holds the left [i,j] point of an edge if horizontal, the bottom [i,j] point if the edge is vertical, and the lower-left [i,j] point if the edge is a diagonal.  It also stores "Horizontal", "Vertical", or "Diagonal" as the second piece of information
    ListSize = len(Clist)
    while ListSize > 0:
        stpoint = random.randint(0,ListSize-1)  #the random point that we will use to find an edge to add to our collection
        #print("List Size", ListSize)
        #print("start point", stpoint)
        
        actualpoint = 0
        counter = -1
        for i in range(len(Clist)):
            if not Clist[i][0] == 0:
                counter += 1
            if counter >= stpoint:
                actualpoint = i
                break
        
        #now the actual point is actualpoint.  We want to randomly choose the available neighboring point
        #first find the set of available points
        #print("Actual Point",actualpoint)
        #print("Clist",Clist)
        avpoints = []
        for i in range(6):
            if not Clist[actualpoint][1][i] is None:
                avpoints.append(i)
        randavpta = 0
        #print("avpoints",avpoints)
        #print(len(avpoints))
        if len(avpoints) > 1:
            randavpta = random.randint(0,len(avpoints)-1)
        randavptb = avpoints[randavpta]
        randavptc = Clist[actualpoint][1][randavptb]
        #now get the [i,j] coordinates of actualpoint
        jpt = actualpoint%N
        ipt = (actualpoint-jpt)//N
        if randavptb == 0: #right
            Elist.append([[ipt,jpt],"Horizontal"])
        elif randavptb == 1:  #left
            Elist.append([[ipt,(N+jpt-1)%N],"Horizontal"])
        elif randavptb == 2:  #up
            Elist.append([[ipt,jpt],"Vertical"])
        elif randavptb == 3:  #down
            Elist.append([[(M+ipt-1)%M,jpt],"Vertical"])
        elif randavptb == 4:  #up-right
            Elist.append([[ipt,jpt],"Diagonal"])    
        else:  #down-left
            Elist.append([[(M+ipt-1)%M,(N+jpt-1)%N],"Diagonal"])    
        #now we clean up Clist
        DualList = [1,0,3,2,5,4]
        for i in range(6):
            if not Clist[actualpoint][1][i] is None:
                Clist[Clist[actualpoint][1][i]][0] -= 1
                Clist[Clist[actualpoint][1][i]][1][DualList[i]] = None
        Clist[actualpoint][0] = 0
        Clist[actualpoint][1] = [None,None,None,None,None,None]
        #now do this for randavptc
        for i in range(6):
            if not Clist[randavptc][1][i] is None:
                Clist[Clist[randavptc][1][i]][0] -= 1
                Clist[Clist[randavptc][1][i]][1][DualList[i]] = None
        Clist[randavptc][0] = 0
        Clist[randavptc][1] = [None,None,None,None,None,None]
        #now find the new ListSize
        ListSize = 0
        for i in range(len(Clist)):
            if Clist[i][0] > 0:
                ListSize += 1
        #print(Elist)
    return Elist    
    
    
        
#Now for the braid 
def MaximalMatchingBraid(Tsize):
    Mmatching = MaximalMatching(Tsize)
    Bsetout = []
    for i in range(len(Mmatching)):
        swdir = random.randint(0,1)
        if swdir == 0:
            Bsetout.append(BraidGenerator(Mmatching[i][0],Mmatching[i][1], sw = "CCW"))
        else:
            Bsetout.append(BraidGenerator(Mmatching[i][0],Mmatching[i][1], sw = "CW"))
    return Bsetout
        
#same as above, but for the triangular case
def MaximalMatchingBraidTri(Tsize):
    Mmatching = MaximalMatchingTri(Tsize)
    Bsetout = []
    for i in range(len(Mmatching)):
        swdir = random.randint(0,1)
        if swdir == 0:
            Bsetout.append(BraidGenerator(Mmatching[i][0],Mmatching[i][1], sw = "CCW"))
        else:
            Bsetout.append(BraidGenerator(Mmatching[i][0],Mmatching[i][1], sw = "CW"))
    return Bsetout    
    