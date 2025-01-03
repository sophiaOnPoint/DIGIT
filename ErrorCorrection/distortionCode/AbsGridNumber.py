import math
def AGN(point,xline,yline,minD):
    minX=10000
    xGridNum=0
    minY=10000
    yGridNum=0
    for i in range(len(xline)):
        d=float(point[0]-xline[i][0])*xline[i][4]-float(point[1]-xline[i][1])*1
        d=d/math.sqrt(1+math.pow(xline[i][4],2))
        d=abs(d)
        if d<minX:
            minX=d
            xGridNum=i
    for i in range(len(yline)):
        d=float(point[0]-yline[i][0])*yline[i][4]-float(point[1]-yline[i][1])*1
        d = d / math.sqrt(1 + math.pow(yline[i][4], 2))
        d=abs(d)
        if d<minY:
            minY=d
            yGridNum=i
    if (minX>minD) or (minY>minD):
        #print('no near grid, jump over this point.')
        return [100000,100000]
    #m=[xGridNum,yGridNum]
    m=[yGridNum,xGridNum]
    return m