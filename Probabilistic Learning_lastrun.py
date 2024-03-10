#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Sat Mar  9 11:40:39 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'Probabilistic Learning'  # from the Builder filename that created this script
expInfo = {
    'participant': '',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/abiodunaje/probabilistic-learning-task/Probabilistic Learning_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1512, 982], fullscr=False, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units=None
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = None
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome" ---
    text = visual.TextStim(win=win, name='text',
        text='\nPlease follow the instructions on the following screens.\n\n\nTo select a symbol on a given side of the screen, use the keys below:\n\nLeft  side =  C          Right side = M\n\n\nPress any key to continue. \n',
        font='Calibri',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    welc_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Prep_Instruction" ---
    instr1 = visual.TextStim(win=win, name='instr1',
        text='Short Preparation session. \n\nYou will be collecting points.\n\nChoose one symbol either on the left (C) or the right (M). \n\nOne symbol is more likely to give you $0.02.\nThe other symbol is more likely to take away $0.02. \n\nChoose the symbol that is the most likely to give you money and avoid the symbol that makes you lose money.\n\n\nPress any key to continue.\n',
        font='Arial',
        pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    intr0_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "prep" ---
    fixation0 = visual.TextStim(win=win, name='fixation0',
        text='+',
        font='Calibri',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    letterpic0_1 = visual.ImageStim(
        win=win,
        name='letterpic0_1', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(-0.1, 0), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    letterpic0_2 = visual.ImageStim(
        win=win,
        name='letterpic0_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0.1, 0), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    prep_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "prep_feedback" ---
    prep_feedb = visual.TextStim(win=win, name='prep_feedb',
        text='',
        font='Calibri',
        pos=(0, 0), height=0.08, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Prep_over" ---
    prep_outro = visual.TextStim(win=win, name='prep_outro',
        text='The practice session is now finished.\n\nYou may have noticed that one of the symbols was more likely to give you money. \n\nThis was not always the case. \n\nIt was more PROBABLE that one symbol gave you money. \n\n\nPress any key to continue. ',
        font='Calibri',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    outro_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Practice_Instruction" ---
    # Run 'Begin Experiment' code from code_6
    myCount = 0
    
    
    pract_intr = visual.TextStim(win=win, name='pract_intr',
        text='Training Session\n\nYou will see 3 pairs of symbols. \n\nIn each pair, one symbol is more PROBABLE to give you money. \n\nIdentify those symbols and choose them.\nAvoid symbols that make you lose money. \n\nUse buttons:\nleft(C)            right(M)\nto make your choices. \n\n\nPress any key to begin.\n',
        font='Calibri',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    instr1_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial" ---
    # Run 'Begin Experiment' code from code
    from numpy.random import random
    
    #create empty variables
    
    myCount1 = 0
    myCount2 = 0
    myCount3 = 0
    
    #create emoty lists
    
    resp1 = []
    resp2 = []
    resp3 = []
    
    resplist1 = []
    resplist2 = []
    resplist3 = []
    
    Corr1 = []
    Corr2 = []
    Corr3 = []
    
    
    
    fixation1 = visual.TextStim(win=win, name='fixation1',
        text='+',
        font='Calibri',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    letterpic1_1 = visual.ImageStim(
        win=win,
        name='letterpic1_1', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[-0.1,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    letterpic1_2 = visual.ImageStim(
        win=win,
        name='letterpic1_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0.1,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "feedback" ---
    blank_feed = visual.TextStim(win=win, name='blank_feed',
        text=None,
        font='Calibri',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    feedb = visual.TextStim(win=win, name='feedb',
        text='',
        font='Calibri',
        pos=(0, 0), height=0.08, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Practice_over" ---
    instr2 = visual.TextStim(win=win, name='instr2',
        text='The training session is now finished. \n\nPress any key to continue onto the next section.\n',
        font='Calibri',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    instr2_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Test_Instr" ---
    instr3 = visual.TextStim(win=win, name='instr3',
        text='Test Session\n\nYou will see the same symbols again. \n\nNow, in different order and combination. \n\nPick the symbol that is the most PROBABLE to give you money.\nAvoid the symbol that makes you lose money.\n\nUse buttons:\nleft(C)            right(M)\n\nMake your decision as quickly as you can.\nIf you are unsure go with your gut feeling.\n\n\nPress any key to begin.\n',
        font='Calibri',
        pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    instr3_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "test_trial" ---
    # Run 'Begin Experiment' code from code_3
    posLB = []
    negLV = []
    fixation2 = visual.TextStim(win=win, name='fixation2',
        text='+',
        font='Calibri',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    letterpic2_1 = visual.ImageStim(
        win=win,
        name='letterpic2_1', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(-0.1, 0), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    letterpic2_2 = visual.ImageStim(
        win=win,
        name='letterpic2_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0.1, 0), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    test_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "check_instr" ---
    check = visual.TextStim(win=win, name='check',
        text='Well done!\n\nNow think about the symbols you saw. \n\nWhich symbol was the MOST likely to give you money? \n\nWhich symbol was the LEAST likely to give you money? \n\nOn the next screen, use numbers:\n1  2  3  4  5  or  6 on top of the keyboard.\n\nIdentify the symbols of your choice. \n\n\nPress any key to continue.',
        font='Arial',
        pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    check_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Learning_Check" ---
    quest = visual.TextStim(win=win, name='quest',
        text='',
        font='Arial',
        pos=(0, 0.3), height=0.06, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    letter_string_pic = visual.ImageStim(
        win=win,
        name='letter_string_pic', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0, 0.1), size=(0.8, 0.12),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    number_string_pic = visual.ImageStim(
        win=win,
        name='number_string_pic', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0, -0.1), size=(0.8, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    learnt_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "test_over" ---
    t_outro = visual.TextStim(win=win, name='t_outro',
        text='The test phase is now finished. \nYou can take a break before continuing. \n\nOtherwise press any key to continue.',
        font='Calibri',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    outro_resp2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "task_over" ---
    task_outro = visual.TextStim(win=win, name='task_outro',
        text='Thank you, the learning task is now finished.\n\nYou can press Esc to exit the full screen. \n\nYou can then close this window. ',
        font='Calibri',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    outro2_resp = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Welcome.started', globalClock.getTime())
    welc_resp.keys = []
    welc_resp.rt = []
    _welc_resp_allKeys = []
    # keep track of which components have finished
    WelcomeComponents = [text, welc_resp]
    for thisComponent in WelcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *welc_resp* updates
        waitOnFlip = False
        
        # if welc_resp is starting this frame...
        if welc_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welc_resp.frameNStart = frameN  # exact frame index
            welc_resp.tStart = t  # local t and not account for scr refresh
            welc_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welc_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welc_resp.started')
            # update status
            welc_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welc_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welc_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welc_resp.status == STARTED and not waitOnFlip:
            theseKeys = welc_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _welc_resp_allKeys.extend(theseKeys)
            if len(_welc_resp_allKeys):
                welc_resp.keys = _welc_resp_allKeys[-1].name  # just the last key pressed
                welc_resp.rt = _welc_resp_allKeys[-1].rt
                welc_resp.duration = _welc_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WelcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in WelcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Welcome.stopped', globalClock.getTime())
    # check responses
    if welc_resp.keys in ['', [], None]:  # No response was made
        welc_resp.keys = None
    thisExp.addData('welc_resp.keys',welc_resp.keys)
    if welc_resp.keys != None:  # we had a response
        thisExp.addData('welc_resp.rt', welc_resp.rt)
        thisExp.addData('welc_resp.duration', welc_resp.duration)
    thisExp.nextEntry()
    # the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Prep_Instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Prep_Instruction.started', globalClock.getTime())
    intr0_resp.keys = []
    intr0_resp.rt = []
    _intr0_resp_allKeys = []
    # keep track of which components have finished
    Prep_InstructionComponents = [instr1, intr0_resp]
    for thisComponent in Prep_InstructionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Prep_Instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr1* updates
        
        # if instr1 is starting this frame...
        if instr1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr1.frameNStart = frameN  # exact frame index
            instr1.tStart = t  # local t and not account for scr refresh
            instr1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr1.started')
            # update status
            instr1.status = STARTED
            instr1.setAutoDraw(True)
        
        # if instr1 is active this frame...
        if instr1.status == STARTED:
            # update params
            pass
        
        # *intr0_resp* updates
        waitOnFlip = False
        
        # if intr0_resp is starting this frame...
        if intr0_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intr0_resp.frameNStart = frameN  # exact frame index
            intr0_resp.tStart = t  # local t and not account for scr refresh
            intr0_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intr0_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intr0_resp.started')
            # update status
            intr0_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intr0_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intr0_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intr0_resp.status == STARTED and not waitOnFlip:
            theseKeys = intr0_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _intr0_resp_allKeys.extend(theseKeys)
            if len(_intr0_resp_allKeys):
                intr0_resp.keys = _intr0_resp_allKeys[-1].name  # just the last key pressed
                intr0_resp.rt = _intr0_resp_allKeys[-1].rt
                intr0_resp.duration = _intr0_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Prep_InstructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Prep_Instruction" ---
    for thisComponent in Prep_InstructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Prep_Instruction.stopped', globalClock.getTime())
    # check responses
    if intr0_resp.keys in ['', [], None]:  # No response was made
        intr0_resp.keys = None
    thisExp.addData('intr0_resp.keys',intr0_resp.keys)
    if intr0_resp.keys != None:  # we had a response
        thisExp.addData('intr0_resp.rt', intr0_resp.rt)
        thisExp.addData('intr0_resp.duration', intr0_resp.duration)
    thisExp.nextEntry()
    # the Routine "Prep_Instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    preparation = data.TrialHandler(nReps=1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('hiragana7.xlsx'),
        seed=None, name='preparation')
    thisExp.addLoop(preparation)  # add the loop to the experiment
    thisPreparation = preparation.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPreparation.rgb)
    if thisPreparation != None:
        for paramName in thisPreparation:
            globals()[paramName] = thisPreparation[paramName]
    
    for thisPreparation in preparation:
        currentLoop = preparation
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPreparation.rgb)
        if thisPreparation != None:
            for paramName in thisPreparation:
                globals()[paramName] = thisPreparation[paramName]
        
        # --- Prepare to start Routine "prep" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prep.started', globalClock.getTime())
        letterpic0_1.setImage(letter10)
        letterpic0_2.setImage(letter20)
        prep_resp.keys = []
        prep_resp.rt = []
        _prep_resp_allKeys = []
        # keep track of which components have finished
        prepComponents = [fixation0, letterpic0_1, letterpic0_2, prep_resp]
        for thisComponent in prepComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prep" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation0* updates
            
            # if fixation0 is starting this frame...
            if fixation0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation0.frameNStart = frameN  # exact frame index
                fixation0.tStart = t  # local t and not account for scr refresh
                fixation0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation0.started')
                # update status
                fixation0.status = STARTED
                fixation0.setAutoDraw(True)
            
            # if fixation0 is active this frame...
            if fixation0.status == STARTED:
                # update params
                pass
            
            # if fixation0 is stopping this frame...
            if fixation0.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation0.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation0.tStop = t  # not accounting for scr refresh
                    fixation0.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation0.stopped')
                    # update status
                    fixation0.status = FINISHED
                    fixation0.setAutoDraw(False)
            
            # *letterpic0_1* updates
            
            # if letterpic0_1 is starting this frame...
            if letterpic0_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letterpic0_1.frameNStart = frameN  # exact frame index
                letterpic0_1.tStart = t  # local t and not account for scr refresh
                letterpic0_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letterpic0_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letterpic0_1.started')
                # update status
                letterpic0_1.status = STARTED
                letterpic0_1.setAutoDraw(True)
            
            # if letterpic0_1 is active this frame...
            if letterpic0_1.status == STARTED:
                # update params
                pass
            
            # if letterpic0_1 is stopping this frame...
            if letterpic0_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letterpic0_1.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    letterpic0_1.tStop = t  # not accounting for scr refresh
                    letterpic0_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letterpic0_1.stopped')
                    # update status
                    letterpic0_1.status = FINISHED
                    letterpic0_1.setAutoDraw(False)
            
            # *letterpic0_2* updates
            
            # if letterpic0_2 is starting this frame...
            if letterpic0_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letterpic0_2.frameNStart = frameN  # exact frame index
                letterpic0_2.tStart = t  # local t and not account for scr refresh
                letterpic0_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letterpic0_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letterpic0_2.started')
                # update status
                letterpic0_2.status = STARTED
                letterpic0_2.setAutoDraw(True)
            
            # if letterpic0_2 is active this frame...
            if letterpic0_2.status == STARTED:
                # update params
                pass
            
            # if letterpic0_2 is stopping this frame...
            if letterpic0_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letterpic0_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    letterpic0_2.tStop = t  # not accounting for scr refresh
                    letterpic0_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letterpic0_2.stopped')
                    # update status
                    letterpic0_2.status = FINISHED
                    letterpic0_2.setAutoDraw(False)
            
            # *prep_resp* updates
            waitOnFlip = False
            
            # if prep_resp is starting this frame...
            if prep_resp.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                prep_resp.frameNStart = frameN  # exact frame index
                prep_resp.tStart = t  # local t and not account for scr refresh
                prep_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prep_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prep_resp.started')
                # update status
                prep_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prep_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prep_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if prep_resp is stopping this frame...
            if prep_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prep_resp.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    prep_resp.tStop = t  # not accounting for scr refresh
                    prep_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prep_resp.stopped')
                    # update status
                    prep_resp.status = FINISHED
                    prep_resp.status = FINISHED
            if prep_resp.status == STARTED and not waitOnFlip:
                theseKeys = prep_resp.getKeys(keyList=['c','m'], ignoreKeys=["escape"], waitRelease=False)
                _prep_resp_allKeys.extend(theseKeys)
                if len(_prep_resp_allKeys):
                    prep_resp.keys = _prep_resp_allKeys[0].name  # just the first key pressed
                    prep_resp.rt = _prep_resp_allKeys[0].rt
                    prep_resp.duration = _prep_resp_allKeys[0].duration
                    # was this correct?
                    if (prep_resp.keys == str(corrAns0)) or (prep_resp.keys == corrAns0):
                        prep_resp.corr = 1
                    else:
                        prep_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prepComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prep" ---
        for thisComponent in prepComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prep.stopped', globalClock.getTime())
        # check responses
        if prep_resp.keys in ['', [], None]:  # No response was made
            prep_resp.keys = None
            # was no response the correct answer?!
            if str(corrAns0).lower() == 'none':
               prep_resp.corr = 1;  # correct non-response
            else:
               prep_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for preparation (TrialHandler)
        preparation.addData('prep_resp.keys',prep_resp.keys)
        preparation.addData('prep_resp.corr', prep_resp.corr)
        if prep_resp.keys != None:  # we had a response
            preparation.addData('prep_resp.rt', prep_resp.rt)
            preparation.addData('prep_resp.duration', prep_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "prep_feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prep_feedback.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_4
        if prep_resp.corr:
            msg0='+$0.02'
            msgColor='green'
        
        if not prep_resp.corr:
            msg0='-$0.02'
            msgColor='red'
        
        if not prep_resp.keys:
            msg0='No response detected'
        prep_feedb.setColor(msgColor, colorSpace='rgb')
        prep_feedb.setText(msg0)
        # keep track of which components have finished
        prep_feedbackComponents = [prep_feedb]
        for thisComponent in prep_feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prep_feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.6:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prep_feedb* updates
            
            # if prep_feedb is starting this frame...
            if prep_feedb.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prep_feedb.frameNStart = frameN  # exact frame index
                prep_feedb.tStart = t  # local t and not account for scr refresh
                prep_feedb.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prep_feedb, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prep_feedb.started')
                # update status
                prep_feedb.status = STARTED
                prep_feedb.setAutoDraw(True)
            
            # if prep_feedb is active this frame...
            if prep_feedb.status == STARTED:
                # update params
                pass
            
            # if prep_feedb is stopping this frame...
            if prep_feedb.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prep_feedb.tStartRefresh + 0.6-frameTolerance:
                    # keep track of stop time/frame for later
                    prep_feedb.tStop = t  # not accounting for scr refresh
                    prep_feedb.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prep_feedb.stopped')
                    # update status
                    prep_feedb.status = FINISHED
                    prep_feedb.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prep_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prep_feedback" ---
        for thisComponent in prep_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prep_feedback.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.600000)
    # completed 1 repeats of 'preparation'
    
    
    # --- Prepare to start Routine "Prep_over" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Prep_over.started', globalClock.getTime())
    outro_resp.keys = []
    outro_resp.rt = []
    _outro_resp_allKeys = []
    # keep track of which components have finished
    Prep_overComponents = [prep_outro, outro_resp]
    for thisComponent in Prep_overComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Prep_over" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prep_outro* updates
        
        # if prep_outro is starting this frame...
        if prep_outro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prep_outro.frameNStart = frameN  # exact frame index
            prep_outro.tStart = t  # local t and not account for scr refresh
            prep_outro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prep_outro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prep_outro.started')
            # update status
            prep_outro.status = STARTED
            prep_outro.setAutoDraw(True)
        
        # if prep_outro is active this frame...
        if prep_outro.status == STARTED:
            # update params
            pass
        
        # *outro_resp* updates
        waitOnFlip = False
        
        # if outro_resp is starting this frame...
        if outro_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            outro_resp.frameNStart = frameN  # exact frame index
            outro_resp.tStart = t  # local t and not account for scr refresh
            outro_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(outro_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'outro_resp.started')
            # update status
            outro_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(outro_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(outro_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if outro_resp.status == STARTED and not waitOnFlip:
            theseKeys = outro_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _outro_resp_allKeys.extend(theseKeys)
            if len(_outro_resp_allKeys):
                outro_resp.keys = _outro_resp_allKeys[-1].name  # just the last key pressed
                outro_resp.rt = _outro_resp_allKeys[-1].rt
                outro_resp.duration = _outro_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Prep_overComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Prep_over" ---
    for thisComponent in Prep_overComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Prep_over.stopped', globalClock.getTime())
    # check responses
    if outro_resp.keys in ['', [], None]:  # No response was made
        outro_resp.keys = None
    thisExp.addData('outro_resp.keys',outro_resp.keys)
    if outro_resp.keys != None:  # we had a response
        thisExp.addData('outro_resp.rt', outro_resp.rt)
        thisExp.addData('outro_resp.duration', outro_resp.duration)
    thisExp.nextEntry()
    # the Routine "Prep_over" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    letterMaster = data.TrialHandler(nReps=1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('letterchoice.xlsx'),
        seed=None, name='letterMaster')
    thisExp.addLoop(letterMaster)  # add the loop to the experiment
    thisLetterMaster = letterMaster.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLetterMaster.rgb)
    if thisLetterMaster != None:
        for paramName in thisLetterMaster:
            globals()[paramName] = thisLetterMaster[paramName]
    
    for thisLetterMaster in letterMaster:
        currentLoop = letterMaster
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLetterMaster.rgb)
        if thisLetterMaster != None:
            for paramName in thisLetterMaster:
                globals()[paramName] = thisLetterMaster[paramName]
        
        # --- Prepare to start Routine "Practice_Instruction" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Practice_Instruction.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_6
        myCount = myCount + 1
        
        if myCount == 3:
            letterMaster.finished = True
        
        
        
        instr1_resp.keys = []
        instr1_resp.rt = []
        _instr1_resp_allKeys = []
        # keep track of which components have finished
        Practice_InstructionComponents = [pract_intr, instr1_resp]
        for thisComponent in Practice_InstructionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Practice_Instruction" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *pract_intr* updates
            
            # if pract_intr is starting this frame...
            if pract_intr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pract_intr.frameNStart = frameN  # exact frame index
                pract_intr.tStart = t  # local t and not account for scr refresh
                pract_intr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pract_intr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pract_intr.started')
                # update status
                pract_intr.status = STARTED
                pract_intr.setAutoDraw(True)
            
            # if pract_intr is active this frame...
            if pract_intr.status == STARTED:
                # update params
                pass
            
            # *instr1_resp* updates
            waitOnFlip = False
            
            # if instr1_resp is starting this frame...
            if instr1_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr1_resp.frameNStart = frameN  # exact frame index
                instr1_resp.tStart = t  # local t and not account for scr refresh
                instr1_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr1_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr1_resp.started')
                # update status
                instr1_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(instr1_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(instr1_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if instr1_resp.status == STARTED and not waitOnFlip:
                theseKeys = instr1_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _instr1_resp_allKeys.extend(theseKeys)
                if len(_instr1_resp_allKeys):
                    instr1_resp.keys = _instr1_resp_allKeys[0].name  # just the first key pressed
                    instr1_resp.rt = _instr1_resp_allKeys[0].rt
                    instr1_resp.duration = _instr1_resp_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Practice_InstructionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Practice_Instruction" ---
        for thisComponent in Practice_InstructionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Practice_Instruction.stopped', globalClock.getTime())
        # check responses
        if instr1_resp.keys in ['', [], None]:  # No response was made
            instr1_resp.keys = None
        letterMaster.addData('instr1_resp.keys',instr1_resp.keys)
        if instr1_resp.keys != None:  # we had a response
            letterMaster.addData('instr1_resp.rt', instr1_resp.rt)
            letterMaster.addData('instr1_resp.duration', instr1_resp.duration)
        # the Routine "Practice_Instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        practice = data.TrialHandler(nReps=1, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(letterset),
            seed=None, name='practice')
        thisExp.addLoop(practice)  # add the loop to the experiment
        thisPractice = practice.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
        if thisPractice != None:
            for paramName in thisPractice:
                globals()[paramName] = thisPractice[paramName]
        
        for thisPractice in practice:
            currentLoop = practice
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
            if thisPractice != None:
                for paramName in thisPractice:
                    globals()[paramName] = thisPractice[paramName]
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            # Run 'Begin Routine' code from code
            # make a count of trials for each pair of letters (A&B, C&D, E&F)
            if (myCount == 1):
                myCount1 = myCount1 + 1
            else:
                myCount1 = myCount1 + 0
            
            if (myCount == 2):
                myCount2 = myCount2 + 1
            else:
                myCount2 = myCount2 + 0
            
            if (myCount == 3):
                myCount3 = myCount3 + 1
            else:
                myCount3 = myCount3 + 0
            
            #Monitor the number of A, C and E responses in the last 60 trials
            #Separate for all three runs of the procedure
            
            if myCount == 1 and (sum(resplist1[-60:]) > 13 and sum(resplist2[-60:]) > 11 and sum(resplist3[-60:]) > 9 and myCount1 > 59):
                practice.finished = True
            
            if myCount == 2 and (sum(resplist1[-60:]) > 13 and sum(resplist2[-60:]) > 11 and sum(resplist3[-60:]) > 9 and myCount2 > 59):
                practice.finished = True
            
            if myCount == 3 and (sum(resplist1[-60:]) > 13 and sum(resplist2[-60:]) > 11 and sum(resplist3[-60:]) > 9 and myCount3 > 59):
                practice.finished = True
            
            
            jitter = random() * (0.8 - 0.3) + 0.3
            jitter = round(jitter, 1) # round to 1 decimal place
            jitter2 = random() * (1.0-0.5) + 0.5
            letterpic1_1.setImage(letter1)
            letterpic1_2.setImage(letter2)
            resp.keys = []
            resp.rt = []
            _resp_allKeys = []
            # keep track of which components have finished
            trialComponents = [fixation1, letterpic1_1, letterpic1_2, resp]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code
                
                
                
                
                # *fixation1* updates
                
                # if fixation1 is starting this frame...
                if fixation1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation1.frameNStart = frameN  # exact frame index
                    fixation1.tStart = t  # local t and not account for scr refresh
                    fixation1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation1.started')
                    # update status
                    fixation1.status = STARTED
                    fixation1.setAutoDraw(True)
                
                # if fixation1 is active this frame...
                if fixation1.status == STARTED:
                    # update params
                    pass
                
                # if fixation1 is stopping this frame...
                if fixation1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation1.tStartRefresh + jitter-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation1.tStop = t  # not accounting for scr refresh
                        fixation1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation1.stopped')
                        # update status
                        fixation1.status = FINISHED
                        fixation1.setAutoDraw(False)
                
                # *letterpic1_1* updates
                
                # if letterpic1_1 is starting this frame...
                if letterpic1_1.status == NOT_STARTED and tThisFlip >= jitter-frameTolerance:
                    # keep track of start time/frame for later
                    letterpic1_1.frameNStart = frameN  # exact frame index
                    letterpic1_1.tStart = t  # local t and not account for scr refresh
                    letterpic1_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(letterpic1_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letterpic1_1.started')
                    # update status
                    letterpic1_1.status = STARTED
                    letterpic1_1.setAutoDraw(True)
                
                # if letterpic1_1 is active this frame...
                if letterpic1_1.status == STARTED:
                    # update params
                    pass
                
                # if letterpic1_1 is stopping this frame...
                if letterpic1_1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > letterpic1_1.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        letterpic1_1.tStop = t  # not accounting for scr refresh
                        letterpic1_1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'letterpic1_1.stopped')
                        # update status
                        letterpic1_1.status = FINISHED
                        letterpic1_1.setAutoDraw(False)
                
                # *letterpic1_2* updates
                
                # if letterpic1_2 is starting this frame...
                if letterpic1_2.status == NOT_STARTED and tThisFlip >= jitter-frameTolerance:
                    # keep track of start time/frame for later
                    letterpic1_2.frameNStart = frameN  # exact frame index
                    letterpic1_2.tStart = t  # local t and not account for scr refresh
                    letterpic1_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(letterpic1_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letterpic1_2.started')
                    # update status
                    letterpic1_2.status = STARTED
                    letterpic1_2.setAutoDraw(True)
                
                # if letterpic1_2 is active this frame...
                if letterpic1_2.status == STARTED:
                    # update params
                    pass
                
                # if letterpic1_2 is stopping this frame...
                if letterpic1_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > letterpic1_2.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        letterpic1_2.tStop = t  # not accounting for scr refresh
                        letterpic1_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'letterpic1_2.stopped')
                        # update status
                        letterpic1_2.status = FINISHED
                        letterpic1_2.setAutoDraw(False)
                
                # *resp* updates
                waitOnFlip = False
                
                # if resp is starting this frame...
                if resp.status == NOT_STARTED and tThisFlip >= jitter-frameTolerance:
                    # keep track of start time/frame for later
                    resp.frameNStart = frameN  # exact frame index
                    resp.tStart = t  # local t and not account for scr refresh
                    resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'resp.started')
                    # update status
                    resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if resp is stopping this frame...
                if resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > resp.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        resp.tStop = t  # not accounting for scr refresh
                        resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'resp.stopped')
                        # update status
                        resp.status = FINISHED
                        resp.status = FINISHED
                if resp.status == STARTED and not waitOnFlip:
                    theseKeys = resp.getKeys(keyList=['c','m'], ignoreKeys=["escape"], waitRelease=False)
                    _resp_allKeys.extend(theseKeys)
                    if len(_resp_allKeys):
                        resp.keys = _resp_allKeys[0].name  # just the first key pressed
                        resp.rt = _resp_allKeys[0].rt
                        resp.duration = _resp_allKeys[0].duration
                        # was this correct?
                        if (resp.keys == str(corrAns)) or (resp.keys == corrAns):
                            resp.corr = 1
                        else:
                            resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            # Run 'End Routine' code from code
            #count how many times participants chose letters A, C and E
            if (resp.keys == letterA):
                resp1 = 1
            else:
                resp1 = 0
            
            if (resp.keys == letterC):
                resp2 = 1
            else:
                resp2 = 0
            
            if (resp.keys == letterE):
                resp3 = 1
            else:
                resp3 = 0
            
            resplist1.append(resp1)
            resplist2.append(resp2)
            resplist3.append(resp3)
            
            Corr1 = sum(resplist1[-60:])
            Corr2 = sum(resplist2[-60:])
            Corr3 = sum(resplist3[-60:])
            
            
            #add a variable in the output that will record when participants
            #respond with A C or E in the learning phase
            practice.addData('resp1', resp1)
            
            practice.addData('resp2', resp2)
            
            practice.addData('resp3', resp3)
            
            
            practice.addData('response1', resplist1[0:10])
            
            practice.addData('response2', resplist2[0:10])
            
            practice.addData('response3', resplist3[0:10])
            
            #add a variable in the output that will count the 
            # performance on different letter pairs
            practice.addData('Corr1', Corr1)
            
            practice.addData('Corr2', Corr2)
            
            practice.addData('Corr3', Corr3)
            
            practice.addData('jitter', jitter)
            
            #add a variable that will count in the output
            # how many times the learning phase and the testing
            # phase occur
            
            practice.addData('myCount1', myCount1)
            practice.addData('myCount2', myCount2)
            practice.addData('myCount3', myCount3)
            
            practice.addData('myCount', myCount)
            # check responses
            if resp.keys in ['', [], None]:  # No response was made
                resp.keys = None
                # was no response the correct answer?!
                if str(corrAns).lower() == 'none':
                   resp.corr = 1;  # correct non-response
                else:
                   resp.corr = 0;  # failed to respond (incorrectly)
            # store data for practice (TrialHandler)
            practice.addData('resp.keys',resp.keys)
            practice.addData('resp.corr', resp.corr)
            if resp.keys != None:  # we had a response
                practice.addData('resp.rt', resp.rt)
                practice.addData('resp.duration', resp.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "feedback" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('feedback.started', globalClock.getTime())
            # Run 'Begin Routine' code from code_2
            if resp.corr:
                msg='+$0.02'
                msgColor='green'
            
            if not resp.corr:
                msg='-$0.02'
                msgColor='red'
            
            if not resp.keys:
                msg='No response detected'
            feedb.setColor(msgColor, colorSpace='rgb')
            feedb.setText(msg)
            # keep track of which components have finished
            feedbackComponents = [blank_feed, feedb]
            for thisComponent in feedbackComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "feedback" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.95:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blank_feed* updates
                
                # if blank_feed is starting this frame...
                if blank_feed.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blank_feed.frameNStart = frameN  # exact frame index
                    blank_feed.tStart = t  # local t and not account for scr refresh
                    blank_feed.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank_feed, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_feed.started')
                    # update status
                    blank_feed.status = STARTED
                    blank_feed.setAutoDraw(True)
                
                # if blank_feed is active this frame...
                if blank_feed.status == STARTED:
                    # update params
                    pass
                
                # if blank_feed is stopping this frame...
                if blank_feed.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank_feed.tStartRefresh + 0.35-frameTolerance:
                        # keep track of stop time/frame for later
                        blank_feed.tStop = t  # not accounting for scr refresh
                        blank_feed.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_feed.stopped')
                        # update status
                        blank_feed.status = FINISHED
                        blank_feed.setAutoDraw(False)
                
                # *feedb* updates
                
                # if feedb is starting this frame...
                if feedb.status == NOT_STARTED and tThisFlip >= 0.35-frameTolerance:
                    # keep track of start time/frame for later
                    feedb.frameNStart = frameN  # exact frame index
                    feedb.tStart = t  # local t and not account for scr refresh
                    feedb.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(feedb, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedb.started')
                    # update status
                    feedb.status = STARTED
                    feedb.setAutoDraw(True)
                
                # if feedb is active this frame...
                if feedb.status == STARTED:
                    # update params
                    pass
                
                # if feedb is stopping this frame...
                if feedb.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > feedb.tStartRefresh + 0.6-frameTolerance:
                        # keep track of stop time/frame for later
                        feedb.tStop = t  # not accounting for scr refresh
                        feedb.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'feedb.stopped')
                        # update status
                        feedb.status = FINISHED
                        feedb.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in feedbackComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "feedback" ---
            for thisComponent in feedbackComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('feedback.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.950000)
        # completed 1 repeats of 'practice'
        
        
        # --- Prepare to start Routine "Practice_over" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Practice_over.started', globalClock.getTime())
        instr2_resp.keys = []
        instr2_resp.rt = []
        _instr2_resp_allKeys = []
        # keep track of which components have finished
        Practice_overComponents = [instr2, instr2_resp]
        for thisComponent in Practice_overComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Practice_over" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *instr2* updates
            
            # if instr2 is starting this frame...
            if instr2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr2.frameNStart = frameN  # exact frame index
                instr2.tStart = t  # local t and not account for scr refresh
                instr2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr2.started')
                # update status
                instr2.status = STARTED
                instr2.setAutoDraw(True)
            
            # if instr2 is active this frame...
            if instr2.status == STARTED:
                # update params
                pass
            
            # *instr2_resp* updates
            waitOnFlip = False
            
            # if instr2_resp is starting this frame...
            if instr2_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr2_resp.frameNStart = frameN  # exact frame index
                instr2_resp.tStart = t  # local t and not account for scr refresh
                instr2_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr2_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr2_resp.started')
                # update status
                instr2_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(instr2_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(instr2_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if instr2_resp.status == STARTED and not waitOnFlip:
                theseKeys = instr2_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _instr2_resp_allKeys.extend(theseKeys)
                if len(_instr2_resp_allKeys):
                    instr2_resp.keys = _instr2_resp_allKeys[0].name  # just the first key pressed
                    instr2_resp.rt = _instr2_resp_allKeys[0].rt
                    instr2_resp.duration = _instr2_resp_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Practice_overComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Practice_over" ---
        for thisComponent in Practice_overComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Practice_over.stopped', globalClock.getTime())
        # check responses
        if instr2_resp.keys in ['', [], None]:  # No response was made
            instr2_resp.keys = None
        letterMaster.addData('instr2_resp.keys',instr2_resp.keys)
        if instr2_resp.keys != None:  # we had a response
            letterMaster.addData('instr2_resp.rt', instr2_resp.rt)
            letterMaster.addData('instr2_resp.duration', instr2_resp.duration)
        # the Routine "Practice_over" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Test_Instr" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Test_Instr.started', globalClock.getTime())
        instr3_resp.keys = []
        instr3_resp.rt = []
        _instr3_resp_allKeys = []
        # keep track of which components have finished
        Test_InstrComponents = [instr3, instr3_resp]
        for thisComponent in Test_InstrComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Test_Instr" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *instr3* updates
            
            # if instr3 is starting this frame...
            if instr3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr3.frameNStart = frameN  # exact frame index
                instr3.tStart = t  # local t and not account for scr refresh
                instr3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr3.started')
                # update status
                instr3.status = STARTED
                instr3.setAutoDraw(True)
            
            # if instr3 is active this frame...
            if instr3.status == STARTED:
                # update params
                pass
            
            # *instr3_resp* updates
            waitOnFlip = False
            
            # if instr3_resp is starting this frame...
            if instr3_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr3_resp.frameNStart = frameN  # exact frame index
                instr3_resp.tStart = t  # local t and not account for scr refresh
                instr3_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr3_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr3_resp.started')
                # update status
                instr3_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(instr3_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(instr3_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if instr3_resp.status == STARTED and not waitOnFlip:
                theseKeys = instr3_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _instr3_resp_allKeys.extend(theseKeys)
                if len(_instr3_resp_allKeys):
                    instr3_resp.keys = _instr3_resp_allKeys[-1].name  # just the last key pressed
                    instr3_resp.rt = _instr3_resp_allKeys[-1].rt
                    instr3_resp.duration = _instr3_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Test_InstrComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Test_Instr" ---
        for thisComponent in Test_InstrComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Test_Instr.stopped', globalClock.getTime())
        # check responses
        if instr3_resp.keys in ['', [], None]:  # No response was made
            instr3_resp.keys = None
        letterMaster.addData('instr3_resp.keys',instr3_resp.keys)
        if instr3_resp.keys != None:  # we had a response
            letterMaster.addData('instr3_resp.rt', instr3_resp.rt)
            letterMaster.addData('instr3_resp.duration', instr3_resp.duration)
        # the Routine "Test_Instr" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        test = data.TrialHandler(nReps=1, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(letterset2),
            seed=None, name='test')
        thisExp.addLoop(test)  # add the loop to the experiment
        thisTest = test.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTest.rgb)
        if thisTest != None:
            for paramName in thisTest:
                globals()[paramName] = thisTest[paramName]
        
        for thisTest in test:
            currentLoop = test
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTest.rgb)
            if thisTest != None:
                for paramName in thisTest:
                    globals()[paramName] = thisTest[paramName]
            
            # --- Prepare to start Routine "test_trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('test_trial.started', globalClock.getTime())
            # Run 'Begin Routine' code from code_3
            jitter = random() * (0.8 - 0.3) + 0.3
            jitter = round(jitter, 1) # round to 1 decimal place
            jitter2 = random() * (1.0-0.5) + 0.5
            letterpic2_1.setImage(letter3)
            letterpic2_2.setImage(letter4)
            test_resp.keys = []
            test_resp.rt = []
            _test_resp_allKeys = []
            # keep track of which components have finished
            test_trialComponents = [fixation2, letterpic2_1, letterpic2_2, test_resp]
            for thisComponent in test_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "test_trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation2* updates
                
                # if fixation2 is starting this frame...
                if fixation2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation2.frameNStart = frameN  # exact frame index
                    fixation2.tStart = t  # local t and not account for scr refresh
                    fixation2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation2.started')
                    # update status
                    fixation2.status = STARTED
                    fixation2.setAutoDraw(True)
                
                # if fixation2 is active this frame...
                if fixation2.status == STARTED:
                    # update params
                    pass
                
                # if fixation2 is stopping this frame...
                if fixation2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation2.tStartRefresh + jitter2-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation2.tStop = t  # not accounting for scr refresh
                        fixation2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation2.stopped')
                        # update status
                        fixation2.status = FINISHED
                        fixation2.setAutoDraw(False)
                
                # *letterpic2_1* updates
                
                # if letterpic2_1 is starting this frame...
                if letterpic2_1.status == NOT_STARTED and tThisFlip >= jitter2-frameTolerance:
                    # keep track of start time/frame for later
                    letterpic2_1.frameNStart = frameN  # exact frame index
                    letterpic2_1.tStart = t  # local t and not account for scr refresh
                    letterpic2_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(letterpic2_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letterpic2_1.started')
                    # update status
                    letterpic2_1.status = STARTED
                    letterpic2_1.setAutoDraw(True)
                
                # if letterpic2_1 is active this frame...
                if letterpic2_1.status == STARTED:
                    # update params
                    pass
                
                # if letterpic2_1 is stopping this frame...
                if letterpic2_1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > letterpic2_1.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        letterpic2_1.tStop = t  # not accounting for scr refresh
                        letterpic2_1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'letterpic2_1.stopped')
                        # update status
                        letterpic2_1.status = FINISHED
                        letterpic2_1.setAutoDraw(False)
                
                # *letterpic2_2* updates
                
                # if letterpic2_2 is starting this frame...
                if letterpic2_2.status == NOT_STARTED and tThisFlip >= jitter2-frameTolerance:
                    # keep track of start time/frame for later
                    letterpic2_2.frameNStart = frameN  # exact frame index
                    letterpic2_2.tStart = t  # local t and not account for scr refresh
                    letterpic2_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(letterpic2_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letterpic2_2.started')
                    # update status
                    letterpic2_2.status = STARTED
                    letterpic2_2.setAutoDraw(True)
                
                # if letterpic2_2 is active this frame...
                if letterpic2_2.status == STARTED:
                    # update params
                    pass
                
                # if letterpic2_2 is stopping this frame...
                if letterpic2_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > letterpic2_2.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        letterpic2_2.tStop = t  # not accounting for scr refresh
                        letterpic2_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'letterpic2_2.stopped')
                        # update status
                        letterpic2_2.status = FINISHED
                        letterpic2_2.setAutoDraw(False)
                
                # *test_resp* updates
                waitOnFlip = False
                
                # if test_resp is starting this frame...
                if test_resp.status == NOT_STARTED and tThisFlip >= jitter2-frameTolerance:
                    # keep track of start time/frame for later
                    test_resp.frameNStart = frameN  # exact frame index
                    test_resp.tStart = t  # local t and not account for scr refresh
                    test_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_resp.started')
                    # update status
                    test_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(test_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(test_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if test_resp is stopping this frame...
                if test_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > test_resp.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        test_resp.tStop = t  # not accounting for scr refresh
                        test_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'test_resp.stopped')
                        # update status
                        test_resp.status = FINISHED
                        test_resp.status = FINISHED
                if test_resp.status == STARTED and not waitOnFlip:
                    theseKeys = test_resp.getKeys(keyList=['c','m'], ignoreKeys=["escape"], waitRelease=False)
                    _test_resp_allKeys.extend(theseKeys)
                    if len(_test_resp_allKeys):
                        test_resp.keys = _test_resp_allKeys[0].name  # just the first key pressed
                        test_resp.rt = _test_resp_allKeys[0].rt
                        test_resp.duration = _test_resp_allKeys[0].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in test_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "test_trial" ---
            for thisComponent in test_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('test_trial.stopped', globalClock.getTime())
            # Run 'End Routine' code from code_3
            
            
            if (test_resp.keys == pickA):
                posLB = 1
            else:
                posLB = 0
            
            if (test_resp.keys == avoidB):
                negLB = 1
            else:
                negLB = 0
            
            
            test.addData('jitter', jitter)
            
            
            test.addData('posLB', posLB)
            test.addData('negLB', negLB)
            
            if myCount > 3:
                letterMaster.finished = True
            # check responses
            if test_resp.keys in ['', [], None]:  # No response was made
                test_resp.keys = None
            test.addData('test_resp.keys',test_resp.keys)
            if test_resp.keys != None:  # we had a response
                test.addData('test_resp.rt', test_resp.rt)
                test.addData('test_resp.duration', test_resp.duration)
            # the Routine "test_trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1 repeats of 'test'
        
        
        # --- Prepare to start Routine "check_instr" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('check_instr.started', globalClock.getTime())
        check_resp.keys = []
        check_resp.rt = []
        _check_resp_allKeys = []
        # keep track of which components have finished
        check_instrComponents = [check, check_resp]
        for thisComponent in check_instrComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "check_instr" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *check* updates
            
            # if check is starting this frame...
            if check.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check.frameNStart = frameN  # exact frame index
                check.tStart = t  # local t and not account for scr refresh
                check.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check.started')
                # update status
                check.status = STARTED
                check.setAutoDraw(True)
            
            # if check is active this frame...
            if check.status == STARTED:
                # update params
                pass
            
            # *check_resp* updates
            waitOnFlip = False
            
            # if check_resp is starting this frame...
            if check_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check_resp.frameNStart = frameN  # exact frame index
                check_resp.tStart = t  # local t and not account for scr refresh
                check_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check_resp.started')
                # update status
                check_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(check_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(check_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if check_resp.status == STARTED and not waitOnFlip:
                theseKeys = check_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _check_resp_allKeys.extend(theseKeys)
                if len(_check_resp_allKeys):
                    check_resp.keys = _check_resp_allKeys[-1].name  # just the last key pressed
                    check_resp.rt = _check_resp_allKeys[-1].rt
                    check_resp.duration = _check_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in check_instrComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "check_instr" ---
        for thisComponent in check_instrComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('check_instr.stopped', globalClock.getTime())
        # check responses
        if check_resp.keys in ['', [], None]:  # No response was made
            check_resp.keys = None
        letterMaster.addData('check_resp.keys',check_resp.keys)
        if check_resp.keys != None:  # we had a response
            letterMaster.addData('check_resp.rt', check_resp.rt)
            letterMaster.addData('check_resp.duration', check_resp.duration)
        # the Routine "check_instr" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        checks = data.TrialHandler(nReps=1, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(letterset3),
            seed=None, name='checks')
        thisExp.addLoop(checks)  # add the loop to the experiment
        thisCheck = checks.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisCheck.rgb)
        if thisCheck != None:
            for paramName in thisCheck:
                globals()[paramName] = thisCheck[paramName]
        
        for thisCheck in checks:
            currentLoop = checks
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisCheck.rgb)
            if thisCheck != None:
                for paramName in thisCheck:
                    globals()[paramName] = thisCheck[paramName]
            
            # --- Prepare to start Routine "Learning_Check" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Learning_Check.started', globalClock.getTime())
            quest.setText(question)
            letter_string_pic.setImage(learnt_letters)
            number_string_pic.setImage(numbers)
            learnt_resp.keys = []
            learnt_resp.rt = []
            _learnt_resp_allKeys = []
            # keep track of which components have finished
            Learning_CheckComponents = [quest, letter_string_pic, number_string_pic, learnt_resp]
            for thisComponent in Learning_CheckComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Learning_Check" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *quest* updates
                
                # if quest is starting this frame...
                if quest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    quest.frameNStart = frameN  # exact frame index
                    quest.tStart = t  # local t and not account for scr refresh
                    quest.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(quest, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'quest.started')
                    # update status
                    quest.status = STARTED
                    quest.setAutoDraw(True)
                
                # if quest is active this frame...
                if quest.status == STARTED:
                    # update params
                    pass
                
                # *letter_string_pic* updates
                
                # if letter_string_pic is starting this frame...
                if letter_string_pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    letter_string_pic.frameNStart = frameN  # exact frame index
                    letter_string_pic.tStart = t  # local t and not account for scr refresh
                    letter_string_pic.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(letter_string_pic, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter_string_pic.started')
                    # update status
                    letter_string_pic.status = STARTED
                    letter_string_pic.setAutoDraw(True)
                
                # if letter_string_pic is active this frame...
                if letter_string_pic.status == STARTED:
                    # update params
                    pass
                
                # *number_string_pic* updates
                
                # if number_string_pic is starting this frame...
                if number_string_pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    number_string_pic.frameNStart = frameN  # exact frame index
                    number_string_pic.tStart = t  # local t and not account for scr refresh
                    number_string_pic.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(number_string_pic, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'number_string_pic.started')
                    # update status
                    number_string_pic.status = STARTED
                    number_string_pic.setAutoDraw(True)
                
                # if number_string_pic is active this frame...
                if number_string_pic.status == STARTED:
                    # update params
                    pass
                
                # *learnt_resp* updates
                waitOnFlip = False
                
                # if learnt_resp is starting this frame...
                if learnt_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    learnt_resp.frameNStart = frameN  # exact frame index
                    learnt_resp.tStart = t  # local t and not account for scr refresh
                    learnt_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(learnt_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'learnt_resp.started')
                    # update status
                    learnt_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(learnt_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(learnt_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if learnt_resp.status == STARTED and not waitOnFlip:
                    theseKeys = learnt_resp.getKeys(keyList=['1','2','3','4','5','6'], ignoreKeys=["escape"], waitRelease=False)
                    _learnt_resp_allKeys.extend(theseKeys)
                    if len(_learnt_resp_allKeys):
                        learnt_resp.keys = _learnt_resp_allKeys[-1].name  # just the last key pressed
                        learnt_resp.rt = _learnt_resp_allKeys[-1].rt
                        learnt_resp.duration = _learnt_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Learning_CheckComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Learning_Check" ---
            for thisComponent in Learning_CheckComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Learning_Check.stopped', globalClock.getTime())
            # check responses
            if learnt_resp.keys in ['', [], None]:  # No response was made
                learnt_resp.keys = None
            checks.addData('learnt_resp.keys',learnt_resp.keys)
            if learnt_resp.keys != None:  # we had a response
                checks.addData('learnt_resp.rt', learnt_resp.rt)
                checks.addData('learnt_resp.duration', learnt_resp.duration)
            # the Routine "Learning_Check" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1 repeats of 'checks'
        
        
        # --- Prepare to start Routine "test_over" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('test_over.started', globalClock.getTime())
        outro_resp2.keys = []
        outro_resp2.rt = []
        _outro_resp2_allKeys = []
        # keep track of which components have finished
        test_overComponents = [t_outro, outro_resp2]
        for thisComponent in test_overComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test_over" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *t_outro* updates
            
            # if t_outro is starting this frame...
            if t_outro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_outro.frameNStart = frameN  # exact frame index
                t_outro.tStart = t  # local t and not account for scr refresh
                t_outro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_outro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_outro.started')
                # update status
                t_outro.status = STARTED
                t_outro.setAutoDraw(True)
            
            # if t_outro is active this frame...
            if t_outro.status == STARTED:
                # update params
                pass
            
            # *outro_resp2* updates
            waitOnFlip = False
            
            # if outro_resp2 is starting this frame...
            if outro_resp2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                outro_resp2.frameNStart = frameN  # exact frame index
                outro_resp2.tStart = t  # local t and not account for scr refresh
                outro_resp2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(outro_resp2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'outro_resp2.started')
                # update status
                outro_resp2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(outro_resp2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(outro_resp2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if outro_resp2.status == STARTED and not waitOnFlip:
                theseKeys = outro_resp2.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _outro_resp2_allKeys.extend(theseKeys)
                if len(_outro_resp2_allKeys):
                    outro_resp2.keys = _outro_resp2_allKeys[-1].name  # just the last key pressed
                    outro_resp2.rt = _outro_resp2_allKeys[-1].rt
                    outro_resp2.duration = _outro_resp2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in test_overComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test_over" ---
        for thisComponent in test_overComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('test_over.stopped', globalClock.getTime())
        # check responses
        if outro_resp2.keys in ['', [], None]:  # No response was made
            outro_resp2.keys = None
        letterMaster.addData('outro_resp2.keys',outro_resp2.keys)
        if outro_resp2.keys != None:  # we had a response
            letterMaster.addData('outro_resp2.rt', outro_resp2.rt)
            letterMaster.addData('outro_resp2.duration', outro_resp2.duration)
        # the Routine "test_over" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1 repeats of 'letterMaster'
    
    
    # --- Prepare to start Routine "task_over" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_over.started', globalClock.getTime())
    outro2_resp.keys = []
    outro2_resp.rt = []
    _outro2_resp_allKeys = []
    # keep track of which components have finished
    task_overComponents = [task_outro, outro2_resp]
    for thisComponent in task_overComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "task_over" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *task_outro* updates
        
        # if task_outro is starting this frame...
        if task_outro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            task_outro.frameNStart = frameN  # exact frame index
            task_outro.tStart = t  # local t and not account for scr refresh
            task_outro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(task_outro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'task_outro.started')
            # update status
            task_outro.status = STARTED
            task_outro.setAutoDraw(True)
        
        # if task_outro is active this frame...
        if task_outro.status == STARTED:
            # update params
            pass
        
        # *outro2_resp* updates
        waitOnFlip = False
        
        # if outro2_resp is starting this frame...
        if outro2_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            outro2_resp.frameNStart = frameN  # exact frame index
            outro2_resp.tStart = t  # local t and not account for scr refresh
            outro2_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(outro2_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'outro2_resp.started')
            # update status
            outro2_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(outro2_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(outro2_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if outro2_resp.status == STARTED and not waitOnFlip:
            theseKeys = outro2_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _outro2_resp_allKeys.extend(theseKeys)
            if len(_outro2_resp_allKeys):
                outro2_resp.keys = _outro2_resp_allKeys[0].name  # just the first key pressed
                outro2_resp.rt = _outro2_resp_allKeys[0].rt
                outro2_resp.duration = _outro2_resp_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_overComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_over" ---
    for thisComponent in task_overComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_over.stopped', globalClock.getTime())
    # check responses
    if outro2_resp.keys in ['', [], None]:  # No response was made
        outro2_resp.keys = None
    thisExp.addData('outro2_resp.keys',outro2_resp.keys)
    if outro2_resp.keys != None:  # we had a response
        thisExp.addData('outro2_resp.rt', outro2_resp.rt)
        thisExp.addData('outro2_resp.duration', outro2_resp.duration)
    thisExp.nextEntry()
    # the Routine "task_over" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
