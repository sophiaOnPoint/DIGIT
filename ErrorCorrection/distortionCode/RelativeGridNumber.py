import AbsGridNumber as AGN
def RGN(point, midpoint, xline, yline, minD):
    xp,yp=AGN.AGN(point,xline, yline, minD)
    xm,ym=AGN.AGN(midpoint,xline,yline,minD)
    return [xp-xm,yp-ym]
