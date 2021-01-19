
import numpy as np
import pdb
import numpy.linalg as la
from numpy import tan, arctan, cos, sin, pi
# import rospy
# from l4vehicle_msgs.msg import VehicleState
# from simulation_control.msg import simulatorStates, errorStates
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# class SimulatorClass():
#     """ Object collecting simulator measurement data:
#         This class listen the published simulator state topic which consists on the vehicle state vector.
#     """

#     def __init__(self):

#         rospy.Subscriber('simulatorStates', simulatorStates, self.simulator_callback, queue_size=1)

#         # Simulator measurement
#         self.x      = 0.0
#         self.y      = 0.0
#         self.psi    = 0.0
#         # self.vx     = rospy.get_param("init_vx")
#         self.vx     = 0.1
#         self.vy     = 0.0
#         self.psiDot = 0.0
#         self.EstimatedData = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#         # simulator_flag = False


#     def simulator_callback(self,data):
#         """Unpack message from sensor, IMU"""

#         # simulator_flag = True
#         self.x       = data.x
#         self.y       = data.y
#         self.psi     = data.psi
#         self.vx      = data.vx
#         self.vy      = data.vy
#         self.psiDot  = data.psiDot
#         self.EstimatedData = [data.vx, data.vy, data.psiDot, data.x, data.y, data.psi]

class Map():
    """map object
    Attributes:
        getGlobalPosition: convert position from (s, ey) to (X,Y)
    """
    def __init__(self, flagTrackShape = 0):
        """Initialization
        Modify the vector spec to change the geometry of the track
        """

        """ Nos interesa que el planner tenga una pista algo mas reducida de la real
        para conservar algo de robustez y no salirnos de la pista en el primer segundo. """
        
        ### is HW is the half width of vehicle dimension + some saftey factor?
        ### what is slack??
        
        # HW            = rospy.get_param("halfWidth")+0.1
        HW = 0.4
        # print ("HW",HW)
        # if flagTrackShape == 0:
        #     selectedTrack = rospy.get_param("trackShape") # comentado para el testeo del planner
        #     # selectedTrack = "L_shape"
        # else:
        #     selectedTrack = "oval"

        selectedTrack = "L_shape"
        print ("track selected",selectedTrack)
        if selectedTrack == "3110":
            self.halfWidth = 0.6
            self.slack     = 0.15
            spec = np.array([[60 * 0.03, 0],
                             [80 * 0.03, +80 * 0.03 * 2 / np.pi],
                             [20 * 0.03, 0],
                             [80 * 0.03, +80 * 0.03 * 2 / np.pi],
                             [40 * 0.03, -40 * 0.03 * 10 / np.pi],
                             [60 * 0.03, +60 * 0.03 * 5 / np.pi],
                             [40 * 0.03, -40 * 0.03 * 10 / np.pi],
                             [80 * 0.03, +80 * 0.03 * 2 / np.pi],
                             [20 * 0.03, 0],
                             [80 * 0.03, +80 * 0.03 * 2 / np.pi],
                             [80 * 0.03, 0]])

        elif selectedTrack == "oval":
            self.halfWidth  = HW
            self.slack      = 0.15
            spec = np.array([[1.0, 0],
                             [4.5, 4.5 / np.pi],
                             [2.0, 0],
                             [4.5, 4.5 / np.pi],
                             [1.0, 0]])

        # elif selectedTrack == "L_shape":
        #     self.halfWidth  = HW
        #     self.slack      = 0.01
        #     lengthCurve     = 4.5
        #     spec = np.array([[1.0, 0],
        #                      [lengthCurve, lengthCurve / np.pi],
        #                      # Note s = 1 * np.pi / 2 and r = -1 ---> Angle spanned = np.pi / 2
        #                      [lengthCurve/2,-lengthCurve / np.pi ],
        #                      [lengthCurve, lengthCurve / np.pi],
        #                      [lengthCurve / np.pi *2, 0],
        #                      [lengthCurve/2, lengthCurve / np.pi]])

        elif selectedTrack == "L_shape_n":
            self.halfWidth  = HW
            self.slack     = 0.01
            lengthCurve     = 4.5
            spec = np.array([[1.0, 0],
                             [lengthCurve, lengthCurve / np.pi],
                             [lengthCurve/2,-lengthCurve / np.pi ],
                             [lengthCurve, lengthCurve / np.pi],
                             [lengthCurve / np.pi *2, 0],
                             [lengthCurve/2, lengthCurve / np.pi]])

        elif selectedTrack == "L_shape_IDIADA":
            self.halfWidth  = HW
            self.slack      = 6*0.45
            lengthCurve     = 10*4.5
            spec = np.array([[1.0, 0],
                             [lengthCurve, lengthCurve / np.pi],
                             # Note s = 1 * np.pi / 2 and r = -1 ---> Angle spanned = np.pi / 2
                             [lengthCurve/2,-lengthCurve / np.pi ],
                             [lengthCurve, lengthCurve / np.pi],
                             [lengthCurve / np.pi *2, 0],
                             [lengthCurve/2, lengthCurve / np.pi]])

        elif selectedTrack == "L_shape":
        # elif selectedTrack == "SLAM_shape1":
            self.halfWidth = 0.4
            self.slack     = 0.01
            lengthCurve    = 1.5*(np.pi/2)
            spec = np.array([[2.5,0],
                             [2*lengthCurve,(lengthCurve*2)/np.pi],
                             [lengthCurve,-(lengthCurve*2) / np.pi],
                             [1.0,0],
                             [lengthCurve,lengthCurve*2/np.pi],
                             [2.0,0],
                             [lengthCurve,(lengthCurve*2)/np.pi],
                             [4.0,0],
                             [lengthCurve,(lengthCurve*2)/np.pi],
                             [2.6,0]])


        elif selectedTrack == "8_track":
            self.halfWidth = 0.4
            self.slack     = 0.15
            lengthCurve    = 1.5*(np.pi/2)
            spec = np.array([[0.5,0],
                             [lengthCurve,(lengthCurve*2)/np.pi],
                             [1.0,0],
                             [lengthCurve,-(lengthCurve*2) / np.pi],
                             [lengthCurve,lengthCurve*2/np.pi],
                             [lengthCurve,lengthCurve*2/np.pi],
                             [1.0,0],
                             [lengthCurve,(lengthCurve*2)/np.pi],
                             [lengthCurve,-(lengthCurve*2)/np.pi],
                             [lengthCurve,(lengthCurve*2)/np.pi],
                             [1.0,0],
                             [lengthCurve,lengthCurve*2/np.pi]])



        # Now given the above segments we compute the (x, y) points of the track and the angle of the tangent vector (psi) at
        # these points. For each segment we compute the (x, y, psi) coordinate at the last point of the segment. Furthermore,
        # we compute also the cumulative s at the starting point of the segment at signed curvature
        # PointAndTangent = [x, y, psi, cumulative s, segment length, signed curvature]

        ### what is cumulative s and signed curvature.?

        PointAndTangent = np.zeros((spec.shape[0] + 1, 6))
        for i in range(0, spec.shape[0]):
            if spec[i, 1] == 0.0:              # If the current segment is a straight line
                l = spec[i, 0]                 # Length of the segments
                if i == 0:
                    ang = 0                          # Angle of the tangent vector at the starting point of the segment
                    x = 0 + l * np.cos(ang)          # x coordinate of the last point of the segment
                    y = 0 + l * np.sin(ang)          # y coordinate of the last point of the segment
                else:
                    ang = PointAndTangent[i - 1, 2]                 # Angle of the tangent vector at the starting point of the segment
                    x = PointAndTangent[i-1, 0] + l * np.cos(ang)  # x coordinate of the last point of the segment
                    y = PointAndTangent[i-1, 1] + l * np.sin(ang)  # y coordinate of the last point of the segment
                psi = ang  # Angle of the tangent vector at the last point of the segment

                # # With the above information create the new line
                # if i == 0:
                #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 0])
                # else:
                #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 0])
                #
                # PointAndTangent[i + 1, :] = NewLine  # Write the new info

                if i == 0:
                    NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 0])
                else:
                    NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 0])

                PointAndTangent[i, :] = NewLine  # Write the new info
            else:
                l = spec[i, 0]                 # Length of the segment
                r = spec[i, 1]                 # Radius of curvature


                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                if i == 0:
                    ang = 0                                                      # Angle of the tangent vector at the
                                                                                 # starting point of the segment
                    CenterX = 0 \
                              + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                    CenterY = 0 \
                              + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle
                else:
                    ang = PointAndTangent[i - 1, 2]                              # Angle of the tangent vector at the
                                                                                 # starting point of the segment
                    CenterX = PointAndTangent[i-1, 0] \
                              + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                    CenterY = PointAndTangent[i-1, 1] \
                              + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

                spanAng = l / np.abs(r)  # Angle spanned by the circle
                psi = wrap(ang + spanAng * np.sign(r))  # Angle of the tangent vector at the last point of the segment

                angleNormal = wrap((direction * np.pi / 2 + ang))
                angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))
                x = CenterX + np.abs(r) * np.cos(
                    angle + direction * spanAng)  # x coordinate of the last point of the segment
                y = CenterY + np.abs(r) * np.sin(
                    angle + direction * spanAng)  # y coordinate of the last point of the segment

                # With the above information create the new line
                # plt.plot(CenterX, CenterY, 'bo')
                # plt.plot(x, y, 'ro')

                # if i == 0:
                #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 1 / r])
                # else:
                #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 1 / r])
                #
                # PointAndTangent[i + 1, :] = NewLine  # Write the new info

                if i == 0:
                    NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 1 / r])
                else:
                    NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 1 / r])

                PointAndTangent[i, :] = NewLine  # Write the new info
            # plt.plot(x, y, 'or')

        # Now update info on last point
        # xs = PointAndTangent[PointAndTangent.shape[0] - 2, 0]
        # ys = PointAndTangent[PointAndTangent.shape[0] - 2, 1]
        # xf = PointAndTangent[0, 0]
        # yf = PointAndTangent[0, 1]
        # psif = PointAndTangent[PointAndTangent.shape[0] - 2, 2]
        #
        # # plt.plot(xf, yf, 'or')
        # # plt.show()
        # l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)
        #
        # NewLine = np.array([xf, yf, psif, PointAndTangent[PointAndTangent.shape[0] - 2, 3] + PointAndTangent[
        #     PointAndTangent.shape[0] - 2, 4], l, 0])
        # PointAndTangent[-1, :] = NewLine


        xs = PointAndTangent[-2, 0]
        ys = PointAndTangent[-2, 1]
        xf = 0
        yf = 0
        psif = 0

        # plt.plot(xf, yf, 'or')
        # plt.show()
        l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)

        NewLine = np.array([xf, yf, psif, PointAndTangent[-2, 3] + PointAndTangent[-2, 4], l, 0])
        PointAndTangent[-1, :] = NewLine

        self.PointAndTangent = PointAndTangent
        self.TrackLength = PointAndTangent[-1, 3] + PointAndTangent[-1, 4]

    def getGlobalPosition(self, s, ey):
        """coordinate transformation from curvilinear reference frame (e, ey) to inertial reference frame (X, Y)
        (s, ey): position in the curvilinear reference frame
        """

        ### what is ey?? error in y coordinate of vehicle from the track inertial frame?

        # wrap s along the track
        while (s > self.TrackLength):
            s = s - self.TrackLength

        # Compute the segment in which system is evolving
        PointAndTangent = self.PointAndTangent

        index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
        ##  i = int(np.where(np.squeeze(index))[0])
        i = np.where(np.squeeze(index))[0]

        if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
            # Extract the first final and initial point of the segment
            xf = PointAndTangent[i, 0]
            yf = PointAndTangent[i, 1]
            xs = PointAndTangent[i - 1, 0]
            ys = PointAndTangent[i - 1, 1]
            psi = PointAndTangent[i, 2]

            # Compute the segment length
            deltaL = PointAndTangent[i, 4]
            reltaL = s - PointAndTangent[i, 3]

            # Do the linear combination
            x = (1 - reltaL / deltaL) * xs + reltaL / deltaL * xf + ey * np.cos(psi + np.pi / 2)
            y = (1 - reltaL / deltaL) * ys + reltaL / deltaL * yf + ey * np.sin(psi + np.pi / 2)
            theta = psi
        else:
            r = 1 / PointAndTangent[i, 5]  # Extract curvature
            ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)
            # Compute the center of the arc
            if r >= 0:
                direction = 1
            else:
                direction = -1

            CenterX = PointAndTangent[i - 1, 0] \
                      + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
            CenterY = PointAndTangent[i - 1, 1] \
                      + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

            spanAng = (s - PointAndTangent[i, 3]) / (np.pi * np.abs(r)) * np.pi

            angleNormal = wrap((direction * np.pi / 2 + ang))
            angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))

            x = CenterX + (np.abs(r) - direction * ey) * np.cos(
                angle + direction * spanAng)  # x coordinate of the last point of the segment
            y = CenterY + (np.abs(r) - direction * ey) * np.sin(
                angle + direction * spanAng)  # y coordinate of the last point of the segment
            theta = ang + direction * spanAng

        return x, y, theta



    def getGlobalPosition_Racing(self, ex, ey, xd, yd, psid):
        """coordinate transformation from curvilinear reference frame (ex, ey) to inertial reference frame (X, Y)
        based on inverse of error computation for racing:
            ex      = +(x-xd)*np.cos(psid) + (y-yd)*np.sin(psid)
            ey      = -(x-xd)*np.sin(psid) + (y-yd)*np.cos(psid)
            epsi    = wrap(psi-psid)
        """

        # x = ex*np.cos(psid) - ey*np.sin(psid) + xd
        x = xd
        y = (ey - xd*np.sin(psid) + yd*np.cos(psid) + x*np.sin(psid)) / np.cos(psid)

        return x, y




    def getLocalPosition(self, x, y, psi):
        """coordinate transformation from inertial reference frame (X, Y) to curvilinear reference frame (s, ey)
        (X, Y): position in the inertial reference frame
        """
        PointAndTangent = self.PointAndTangent
        CompletedFlag = 0



        for i in range(0, PointAndTangent.shape[0]):
            if CompletedFlag == 1:
                break

            if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
                # Extract the first final and initial point of the segment
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                psi_unwrap = np.unwrap([PointAndTangent[i - 1, 2], psi])[1]
                epsi = psi_unwrap - PointAndTangent[i - 1, 2]

                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    s  = PointAndTangent[i, 3]
                    ey = 0
                    CompletedFlag = 1

                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    CompletedFlag = 1
                else:
                    if np.abs(computeAngle( [x,y] , [xs, ys], [xf, yf])) <= np.pi/2 and np.abs(computeAngle( [x,y] , [xf, yf], [xs, ys])) <= np.pi/2:
                        v1 = np.array([x,y]) - np.array([xs, ys])
                        angle = computeAngle( [xf,yf] , [xs, ys], [x, y])
                        s_local = la.norm(v1) * np.cos(angle)
                        s       = s_local + PointAndTangent[i, 3]
                        ey      = la.norm(v1) * np.sin(angle)

                        if np.abs(ey)<= self.halfWidth + self.slack:
                            CompletedFlag = 1

            else:
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                r = 1 / PointAndTangent[i, 5]  # Extract curvature
                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)

                # Compute the center of the arc
                CenterX = xs + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                CenterY = ys + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    ey = 0
                    psi_unwrap = np.unwrap([ang, psi])[1]
                    epsi = psi_unwrap - ang
                    s = PointAndTangent[i, 3]
                    CompletedFlag = 1
                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    psi_unwrap = np.unwrap([PointAndTangent[i, 2], psi])[1]
                    epsi = psi_unwrap - PointAndTangent[i, 2]
                    CompletedFlag = 1
                else:
                    arc1 = PointAndTangent[i, 4] * PointAndTangent[i, 5]
                    arc2 = computeAngle([xs, ys], [CenterX, CenterY], [x, y])
                    if np.sign(arc1) == np.sign(arc2) and np.abs(arc1) >= np.abs(arc2):
                        v = np.array([x, y]) - np.array([CenterX, CenterY])
                        s_local = np.abs(arc2)*np.abs(r)
                        s    = s_local + PointAndTangent[i, 3]
                        ey   = -np.sign(direction) * (la.norm(v) - np.abs(r))
                        psi_unwrap = np.unwrap([ang + arc2, psi])[1]
                        epsi = psi_unwrap - (ang + arc2)

                        if np.abs(ey) <= self.halfWidth + self.slack: # OUT OF TRACK!!
                            CompletedFlag = 1

        # if epsi>1.0:
        #     print "epsi Greater then 1.0"
        #     pdb.set_trace()

        if CompletedFlag == 0:
            s    = 10000
            ey   = 10000
            epsi = 10000
            #print "Error!! POINT OUT OF THE TRACK!!!! <=================="
            # pdb.set_trace()

        return s, ey, epsi, CompletedFlag



# ======================================================================================================================
# ======================================================================================================================
# ====================================== Internal utilities functions ==================================================
# ======================================================================================================================
# ======================================================================================================================

def computeAngle(point1, origin, point2):
    # The orientation of this angle matches that of the coordinate system. Tha is why a minus sign is needed
    v1 = np.array(point1) - np.array(origin)
    v2 = np.array(point2) - np.array(origin)

    dot = v1[0] * v2[0] + v1[1] * v2[1]  # dot product between [x1, y1] and [x2, y2]
    det = v1[0] * v2[1] - v1[1] * v2[0]  # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    return angle # np.arctan2(sinang, cosang)


def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


def sign(a):
    if a >= 0:
        res = 1
    else:
        res = -1

    return res


def unityTestChangeOfCoordinates(map, ClosedLoopData):
    """For each point in ClosedLoopData change (X, Y) into (s, ey) and back to (X, Y) to check accurancy
    """
    TestResult = 1
    for i in range(0, ClosedLoopData.x.shape[0]):
        xdat = ClosedLoopData.x
        xglobdat = ClosedLoopData.x_glob

        s, ey, _, _ = map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5], xglobdat[i, 3])
        v1 = np.array([s, ey])
        v2 = np.array(xdat[i, 4:6])
        v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
        v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
        # print v1, v2, np.dot(v1 - v2, v1 - v2), np.dot(v3 - v4, v3 - v4)

        if np.dot(v3 - v4, v3 - v4) > 0.00000001:
            TestResult = 0
            print ("ERROR", v1, v2, v3, v4)
            pdb.set_trace()
            v1 = np.array(map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5]))
            v2 = np.array(xdat[i, 4:6])
            v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
            v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
            print (np.dot(v3 - v4, v3 - v4))
            pdb.set_trace()

    if TestResult == 1:
        print ("Change of coordinates test passed!")


def _initializeFigure_xy(map):
    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    axtr = plt.axes()

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    # Points1 = np.zeros((Points, 2))
    # Points2 = np.zeros((Points, 2))
    # Points0 = np.zeros((Points, 2))
    Points1 = np.zeros((Points, 3))
    Points2 = np.zeros((Points, 3))
    Points0 = np.zeros((Points, 3))    

    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o') #points on center track
    plt.plot(Points0[:, 0], Points0[:, 1], '--') # center line
    # np.save('inner_track',np.array([Points0[:, 0], Points0[:, 1]]))
    plt.plot(Points1[:, 0], Points1[:, 1], '-b') # inner track
    plt.plot(Points2[:, 0], Points2[:, 1], '-b') #outer track



    line_cl,        = axtr.plot(xdata, ydata, '-k')
    line_gps_cl,    = axtr.plot(xdata, ydata, '--ob')  # Plots the traveled positions
    line_tr,        = axtr.plot(xdata, ydata, '-or')       # Plots the current positions
    line_SS,        = axtr.plot(xdata, ydata, 'og')
    line_pred,      = axtr.plot(xdata, ydata, '-or')
    line_planning,  = axtr.plot(xdata, ydata, '-ok')
    
    l = 0.4; w = 0.2 #legth and width of the car

    v = np.array([[ 0.4,  0.2],
                  [ 0.4, -0.2],
                  [-0.4, -0.2],
                  [-0.4,  0.2]])

    rec = patches.Polygon(v, alpha=0.7, closed=True, fc='r', ec='k', zorder=10)
    axtr.add_patch(rec)

    # Vehicle:
    rec_sim = patches.Polygon(v, alpha=0.7, closed=True, fc='G', ec='k', zorder=10)

    # if mode == "simulations":
    #     axtr.add_patch(rec_sim)    

    # Planner vehicle:
    rec_planning = patches.Polygon(v, alpha=0.7, closed=True, fc='k', ec='k', zorder=10)


    # plt.show()
    # plt.pause(2)
    return fig, axtr, line_planning, line_tr, line_pred, line_SS, line_cl, line_gps_cl, rec, rec_sim, rec_planning

def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y

# track_map   = Map()
# ( fig, axtr, line_planning, line_tr, line_pred, line_SS, line_cl, line_gps_cl, rec,\
#          rec_sim, rec_planning ) = _initializeFigure_xy(track_map)

        # Plotting the estimated car: 
    
# x = 0.0
# y = 0.0
# psi = 0 
# l = 0.4; w = 0.2 #legth and width of the car
# line_tr.set_data(x, y)
# car_x, car_y = getCarPosition(x, y, psi, w, l)
# print ("car_x,car_y",car_x,car_y)
# rec.set_xy(np.array([car_x, car_y]).T)      
# s, ey, epsi, insideMap = track_map.getLocalPosition(x, y, psi)
# print ("insideMap",insideMap)
# plt.show()
# plt.pause(2)

# for i in range(100):
#     print('Enter x value:')
#     x = float(input())
#     print('Enter y value:')
#     y = float(input())
#     print('Enter psi value:')
#     psi = float(input())

#     # x = 0.0
#     # y = 4
#     # psi = 0.0 
#     l = 0.4; w = 0.2 #legth and width of the car
#     line_tr.set_data(x, y)
#     car_x, car_y = getCarPosition(x, y, psi, w, l)
#     print ("car_x,car_y",car_x,car_y)
    
#     s, ey, epsi, insideMap = track_map.getLocalPosition(x, y, psi)
#     print ("s",s, "ey",ey, "epsi",(epsi*180.0)/pi,"insideMap",insideMap)

#     rec.set_xy(np.array([car_x, car_y]).T)      
#     plt.show()
#     plt.pause(0.5)


# def kin_sim(z,u,dt):
#         # get states / inputs
#         x       = z[0]
#         y       = z[1]
#         psi     = z[2]
#         v       = z[3]
#         d_f     = u[0]
#         a       = u[1]

#         # extract parameters
#         L_a = 0.125
#         L_b = 0.125

#         # compute slip angle
#         bta         = arctan( L_a / (L_a + L_b) * tan(d_f) )

#         # compute next state
#         x_next      = x + dt*( v*cos(psi + bta) )
#         y_next      = y + dt*( v*sin(psi + bta) )
#         psi_next    = psi + dt*v/L_b*sin(bta)
#         v_next      = v + dt*a

#         return [x_next, y_next, psi_next, v_next]

    
# # steer_in = np.linspace(0,0.254,50)
# # acc_in = np.linspace(0,1,50)
# x_next =0
# y_next =0 
# psi_next = 0 
# v_next = 0
# dt = 0.1
# z = [x_next, y_next, psi_next, v_next]
# # data = []
# # for i,(acc,steer) in enumerate(zip(acc_in,steer_in)):
# #         u = [steer,acc]
# #         z = kin_sim(z,u,dt)
# #         data.append(z)
# # data = np.dstack(data)
# # plt.plot(data[0][0],data[0][1])  


# max_acceleration=3.0 # max m/s2
# max_angle_steering=0.249 #Absolut value
# actions=[]
# for i in np.linspace(-2,max_acceleration,20): # 5 values of acceleration
#     for j in np.linspace(-max_angle_steering,max_angle_steering,7): # / values of angle steering
        
#         # u = [round(i,1),round(j,3)]
#         # z = kin_sim(z,u,dt)
#         # z = [x_next, y_next, psi_next, v_next]
#         # actions.append([round(i,1),round(j,3)])
#         x_next = round(i,1)
#         y_next = round(j,3)
#         line_tr.set_data(x_next, y_next)
#         car_x, car_y = getCarPosition(x_next, y_next, psi_next, w, l)
#         print ("car_x,car_y",car_x,car_y)
#         rec.set_xy(np.array([car_x, car_y]).T)      
#         s, ey, epsi, insideMap = track_map.getLocalPosition(x_next, y_next, psi_next)
#         print ("insideMap",insideMap)
#         plt.show()
#         plt.pause(1)





# def main():

#     rospy.init_node("track_vehicle")
#     loop_rate       = 30.0
#     dt              = 1.0/loop_rate
#     rate            = rospy.Rate(loop_rate)

#     sim = SimulatorClass()
#     pub_errorStates = rospy.Publisher('errorStates', errorStates, queue_size=1)
#     mode        = rospy.get_param("/track_vehicle/mode") # testing planner
#     plotGPS     = rospy.get_param("/track_vehicle/plotGPS")   
#     mode = "cvxcv"
#     # mode = "simulations"
#     # plotGPS = False
#     # data        = Estimation_Mesures_Planning_Data(mode, plotGPS)        
#     track_map   = Map()

#     loop_rate   = 30.0
#     rate        = rospy.Rate(loop_rate)

#     vSwitch     = 1.3     
#     psiSwitch   = 1.2    

#     StateView   = False    

#     if StateView == True:

#     else:

#         Planning_Track = rospy.get_param("/track_vehicle/Planning_Track")
#         ( fig, axtr, line_planning, line_tr, line_pred, line_SS, line_cl, line_gps_cl, rec,
#          rec_sim, rec_planning ) = _initializeFigure_xy(track_map, mode)

#     ClosedLoopTraj_gps_x = []
#     ClosedLoopTraj_gps_y = []
#     ClosedLoopTraj_x = []
#     ClosedLoopTraj_y = []

#     flagHalfLap = False

#     while not rospy.is_shutdown():
#         estimatedStates = sim.EstimatedData
#         err_state = errorStates()
#         s, ey, epsi, insideMap = track_map.getLocalPosition(estimatedStates[3], estimatedStates[4], estimatedStates[5])

#         if s > track_map.TrackLength / 2:
#             flagHalfLap = True

#         if (s < track_map.TrackLength / 4) and (flagHalfLap == True): # New lap
#             ClosedLoopTraj_gps_x = []
#             ClosedLoopTraj_gps_y = []
#             ClosedLoopTraj_x = []
#             ClosedLoopTraj_y = []
#             flagHalfLap = False

#         x = estimatedStates[3]
#         y = estimatedStates[4]

#         if plotGPS == True:
#             ClosedLoopTraj_gps_x.append(data.MeasuredData[0])
#             ClosedLoopTraj_gps_y.append(data.MeasuredData[1])

#         ClosedLoopTraj_x.append(x) 
#         ClosedLoopTraj_y.append(y)

#         psi = estimatedStates[5]
#         l = 0.4; w = 0.2


#         line_cl.set_data(ClosedLoopTraj_x, ClosedLoopTraj_y)

#         if plotGPS == True:
#             line_gps_cl.set_data(ClosedLoopTraj_gps_x, ClosedLoopTraj_gps_y)

#         if StateView == True:
#             linevx.set_data(0.0, estimatedStates[0])
#             linevy.set_data(0.0, estimatedStates[1])
#             linewz.set_data(0.0, estimatedStates[2])
#             lineepsi.set_data(0.0, epsi)
#             lineey.set_data(0.0, ey)

#         # Plotting the estimated car:        
#         line_tr.set_data(estimatedStates[3], estimatedStates[4])
#         car_x, car_y = getCarPosition(x, y, psi, w, l)
#         rec.set_xy(np.array([car_x, car_y]).T)        

#         print "[Car position]\n","X",x,"Y",y,"theta",estimatedStates[-1]*180/3.14
#         print "[Car velocity and angular velocity]\n","dX",estimatedStates[0],"dY",estimatedStates[1],"dtheta",estimatedStates[2]*180/3.14
        
#         print "insideMap",insideMap
#         print "lateral error from the center:",ey,"\n steering error from the center line" ,epsi*180/3.14
        
#         err_state.ey        = ey
#         err_state.epsi      = epsi
#         err_state.insideMap = insideMap

#         # Publish error states
#         pub_errorStates.publish(err_state)

#         if insideMap == 1:
#             fig.canvas.draw()

#         rate.sleep()


# if __name__ == "__main__":

#     try:
#         main()

#     except rospy.ROSInterruptException:
#         pass