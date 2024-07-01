#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

r"""

.. figure:: /_images/static/Vizard1.jpg
   :align: center

   Illustration of Vizard showing a custom spacecraft camera view.

Overview
--------

This scenario demonstrates how instantiate a visualization interface. This includes setting camera
parameters and capture rates. This stems for an attitude detumble scenario, but focuses on
pointing towards a celestial body in order to display the visualization Vizard, and show
the camera capabilities.

The script is found in the folder ``basilisk/examples`` and executed by using::

    python3 scenarioVizPoint.py

When the simulation completes 3 plots are shown for the MRP attitude history, the rate
tracking errors, as well as the control torque vector.  The ``run()`` method is setup to write out the
Vizard data file to sub-folder ``_VizFiles/scenarioVizPoint_UnityViz.bin``.  By running :ref:`Vizard <vizard>`
and playing back this data file you will see the custom camera view that is created as
illustrated in the Vizard snapshot above.

The simulation layout is identical the the :ref:`scenarioAttitudeFeedback` scenario when it comes to FSW modules
The spacecraft starts in a tumble and controls it's rate as well as points to the Earth.

Two mission scenarios can be simulated.
The first one mimics the DSCOVR mission spacecraft and its EPIC camera pointing towards Earth.
The second simulates a spacecraft orbiting about Mars. The attitude results are the same as
the attitude feedback scenario, and pictured in the following plots. The differences lies in
where they are pointing.

.. image:: /_images/Scenarios/scenarioVizPoint1.svg
   :align: center

.. image:: /_images/Scenarios/scenarioVizPoint2.svg
   :align: center

.. image:: /_images/Scenarios/scenarioVizPoint3.svg
   :align: center

In each case a spacecraft fixed camera is simulated.
This is done by connecting to the :ref:`vizInterface` input message
``cameraConfInMsg``  The :ref:`vizInterface` module
checks this input message by default.  If it is linked, then the camera information
is read in and sent across to Vizard to render out that camera view point image.
Open Vizard and play back the resulting simulation binary file to see the camera window.

DSCOVR Mission Setup
--------------------

The first setup has the spacecraft pointing to Earth, from a distant, L1 vantage point.
The scenario controls the spacecraft attitude to Earth pointing mode, and snaps pictures at
a defined rate.
This camera parameters are taken from NASA's `EPIC <https://epic.gsfc.nasa.gov>`__ camera website on the date
2018 OCT 23 04:35:25.000 (UTC time).
In this setup the pointing needs to be set to Earth, given it's position.

Mars Orbit Setup
----------------

The second control scenario points the spacecraft towards Mars on a Mars orbit.

"""


#
# Basilisk Scenario Script and Integrated Test
#
# Purpose:  Integrated test of the vizInterface, spacecraft, simpleNav, mrpFeedback. and inertial3D modules.
# Illustrates a spacecraft pointing with visualization.
# Author:   Thibaud Teil
# Creation Date:  Nov. 01, 2018
#

import os

import numpy as np
from Basilisk import __path__

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])
fileNamePath = os.path.abspath(__file__)


# import general simulation support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
import matplotlib.pyplot as plt
from Basilisk.utilities import macros, orbitalMotion
from Basilisk.utilities import RigidBodyKinematics as rbk

# import simulation related support
from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque
from Basilisk.utilities import simIncludeGravBody
from Basilisk.simulation import simpleNav

# import FSW Algorithm related support
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.fswAlgorithms import attTrackingError

# import message declarations
from Basilisk.architecture import messaging

# attempt to import vizard
from Basilisk.utilities import vizSupport


def run():
   
    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    # set the simulation time variable used later on
    simulationTime = macros.min2nano(10.)

    #
    #  create the simulation process
    #
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(.1)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

   
    simulationTime = macros.min2nano(6.25)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    # setup Earth Gravity Body
    earth = gravFactory.createEarth()
    earth.isCentralBody = True  # ensure this is the central gravitational body
    mu = earth.mu


    spacecraftList = []
    
    # Number of satellites to simulate
    numSatellites = 3

    I = [900., 0., 0., 0., 800., 0., 0., 0., 600.]
    # create the FSW vehicle configuration message
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = I  # use the same inertia in the FSW algorithm as in the simulation
    vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    for i in range(numSatellites):
        # Create a new spacecraft object for each satellite
        scObject = spacecraft.Spacecraft()
        scObject.ModelTag = f"spacecraftBody_{i}"
        
        # Set properties (you can vary these slightly for each satellite if desired)
        I = [900., 0., 0., 0., 800., 0., 0., 0., 600.]
        scObject.hub.mHub = 750.0
        scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
        scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
        
        # Attach gravity model
        gravFactory.addBodiesTo(scObject)
        
        # Add spacecraft object to the simulation process
        scSim.AddModelToTask(simTaskName, scObject)
        
        # Setup extForceTorque module for each satellite
        extFTObject = extForceTorque.ExtForceTorque()
        extFTObject.ModelTag = f"externalDisturbance_{i}"
        scObject.addDynamicEffector(extFTObject)
        scSim.AddModelToTask(simTaskName, extFTObject)
        
        # Add simple Navigation sensor module
        sNavObject = simpleNav.SimpleNav()
        sNavObject.ModelTag = f"SimpleNavigation_{i}"
        scSim.AddModelToTask(simTaskName, sNavObject)
        sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
        
        # Setup FSW modules for each satellite
        inertial3DObj = inertial3D.inertial3D()
        inertial3DObj.ModelTag = f"inertial3D_{i}"
        scSim.AddModelToTask(simTaskName, inertial3DObj)
        inertial3DObj.sigma_R0N = [0., 0., 0.1]
        
        attError = attTrackingError.attTrackingError()
        attError.ModelTag = f"attErrorInertial3D_{i}"
        scSim.AddModelToTask(simTaskName, attError)
        attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
        attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
        
        mrpControl = mrpFeedback.mrpFeedback()
        mrpControl.ModelTag = f"mrpFeedback_{i}"
        scSim.AddModelToTask(simTaskName, mrpControl)
        mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
        mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
        extFTObject.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
        
        # Set initial states for each satellite
        oe = orbitalMotion.ClassicElements()
        oe.a = 16000000 + i * 100000  # Slightly different semi-major axis for each satellite
        oe.e = 0.1
        oe.i = (10. + i) * macros.D2R  # Slightly different inclination
        oe.Omega = (25. + i * 5) * macros.D2R
        oe.omega = 10. * macros.D2R
        oe.f = (160. + i * 10) * macros.D2R
        rN, vN = orbitalMotion.elem2rv(mu, oe)
        
        scObject.hub.r_CN_NInit = rN
        scObject.hub.v_CN_NInit = vN
        scObject.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]
        scObject.hub.omega_BN_BInit = [[0.001], [-0.01], [0.03]]
        
        spacecraftList.append(scObject)

#     #
#     #   setup the simulation tasks/objects
#     #
#     # initialize spacecraft object and set properties
#     scObject = spacecraft.Spacecraft()
#     scObject.ModelTag = "spacecraftBody"
#     # define the simulation inertia
#     I = [900., 0., 0.,
#          0., 800., 0.,
#          0., 0., 600.]
#     scObject.hub.mHub = 750.0  # kg - spacecraft mass
#     scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
#     scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
#     # attach gravity model to spacecraft
#     gravFactory.addBodiesTo(scObject)

#     # add spacecraft object to the simulation process
#     scSim.AddModelToTask(simTaskName, scObject)

#     # setup extForceTorque module
#     # the control torque is read in through the messaging system
#     extFTObject = extForceTorque.ExtForceTorque()
#     extFTObject.ModelTag = "externalDisturbance"
# #    extFTObject.extTorquePntB_B = [[0.25], [-0.25], [0.1]]
#     scObject.addDynamicEffector(extFTObject)
#     scSim.AddModelToTask(simTaskName, extFTObject)

#     # add the simple Navigation sensor module.  This sets the SC attitude, rate, position
#     # velocity navigation message
#     sNavObject = simpleNav.SimpleNav()
#     sNavObject.ModelTag = "SimpleNavigation"
#     scSim.AddModelToTask(simTaskName, sNavObject)
#     sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

#     #
#     #   setup the FSW algorithm tasks
#     #
#     earthPoint = np.array([0.,0.,0.1])

#     # create the FSW vehicle configuration message
#     vehicleConfigOut = messaging.VehicleConfigMsgPayload()
#     vehicleConfigOut.ISCPntB_B = I  # use the same inertia in the FSW algorithm as in the simulation
#     vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

#     # setup inertial3D guidance module
#     inertial3DObj = inertial3D.inertial3D()
#     inertial3DObj.ModelTag = "inertial3D"
#     scSim.AddModelToTask(simTaskName, inertial3DObj)
#     inertial3DObj.sigma_R0N = earthPoint.tolist()  # set the desired inertial orientation

#     # setup the attitude tracking error evaluation module
#     attError = attTrackingError.attTrackingError()
#     attError.ModelTag = "attErrorInertial3D"
#     scSim.AddModelToTask(simTaskName, attError)
#     attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
#     attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)

#     # setup the MRP Feedback control module
#     mrpControl = mrpFeedback.mrpFeedback()
#     mrpControl.ModelTag = "mrpFeedback"
#     scSim.AddModelToTask(simTaskName, mrpControl)
#     mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
#     mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
#     extFTObject.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
#     mrpControl.K = 3.5
#     mrpControl.Ki = -1  # make value negative to turn off integral feedback
#     mrpControl.P = 30.0
#     mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1

#     #
#     #   Setup data logging before the simulation is initialized
#     #
#     numDataPoints = 100
#     samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
#     cmdRec = mrpControl.cmdTorqueOutMsg.recorder(samplingTime)
#     attErrRec = attError.attGuidOutMsg.recorder(samplingTime)
#     dataLog = sNavObject.transOutMsg.recorder(samplingTime)
#     scSim.AddModelToTask(simTaskName, cmdRec)
#     scSim.AddModelToTask(simTaskName, attErrRec)
#     scSim.AddModelToTask(simTaskName, dataLog)

#     #
#     #   set initial Spacecraft States
#     #
#     # setup the orbit using classical orbit elements
#     # for orbit around Earth
#     oe = orbitalMotion.ClassicElements()
#     oe.a = 16000000 # meters
#     oe.e = 0.1
#     oe.i = 10. * macros.D2R
#     oe.Omega = 25. * macros.D2R
#     oe.omega = 10. * macros.D2R
#     oe.f = 160. * macros.D2R
#     rN, vN = orbitalMotion.elem2rv(mu, oe)

#     scObject.hub.r_CN_NInit = rN  # m   - r_CN_N
#     scObject.hub.v_CN_NInit = vN  # m/s - v_CN_N
#     scObject.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]  # sigma_BN_B
#     scObject.hub.omega_BN_BInit = [[0.001], [-0.01], [0.03]]  # rad/s - omega_BN_B

    #
    #   initialize Simulation
    #

    viz = vizSupport.enableUnityVisualization(scSim, simTaskName, spacecraftList,
                                                saveFile=fileNamePath)
    # viz.addCamMsgToModule(camMsg)
    # viz.settings.viewCameraConeHUD = 1
    scSim.InitializeSimulation()

    #
    #   configure a simulation stop time and execute the simulation run
    #
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

   
if __name__ == "__main__":
    run(
    )