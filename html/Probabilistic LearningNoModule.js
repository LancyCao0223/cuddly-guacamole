/******************************* 
 * Probabilistic Learning Test *
 *******************************/

// init psychoJS:
var psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0, 0, 0]),
  units: 'height'
});

// store info about the experiment session:
let expName = 'Probabilistic Learning';  // from the Builder filename that created this script
let expInfo = {'Participant ID*': ''};

// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(WelcomeRoutineBegin);
flowScheduler.add(WelcomeRoutineEachFrame);
flowScheduler.add(WelcomeRoutineEnd);
flowScheduler.add(Prep_InstructionRoutineBegin);
flowScheduler.add(Prep_InstructionRoutineEachFrame);
flowScheduler.add(Prep_InstructionRoutineEnd);
const preparationLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(preparationLoopBegin, preparationLoopScheduler);
flowScheduler.add(preparationLoopScheduler);
flowScheduler.add(preparationLoopEnd);
flowScheduler.add(Prep_overRoutineBegin);
flowScheduler.add(Prep_overRoutineEachFrame);
flowScheduler.add(Prep_overRoutineEnd);
const letterMasterLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(letterMasterLoopBegin, letterMasterLoopScheduler);
flowScheduler.add(letterMasterLoopScheduler);
flowScheduler.add(letterMasterLoopEnd);
flowScheduler.add(task_overRoutineBegin);
flowScheduler.add(task_overRoutineEachFrame);
flowScheduler.add(task_overRoutineEnd);
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({expName, expInfo});

var frameDur;
function updateInfo() {
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '3.1.0';

  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0/Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0/60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  
  return Scheduler.Event.NEXT;
}

var WelcomeClock;
var text;
var Prep_InstructionClock;
var instr1;
var prepClock;
var fixation0;
var letterpic0_1;
var letterpic0_2;
var prep_feedbackClock;
var prep_feedb;
var Prep_overClock;
var prep_outro;
var Practice_InstructionClock;
var myCount;
var pract_intr;
var trialClock;
var myCount1;
var myCount2;
var myCount3;
var resp1;
var resp2;
var resp3;
var resplist1;
var resplist2;
var resplist3;
var Corr1;
var Corr2;
var Corr3;
var fixation1;
var letterpic1_1;
var letterpic1_2;
var feedbackClock;
var blank_feed;
var feedb;
var Practice_overClock;
var instr2;
var Test_InstrClock;
var instr3;
var test_trialClock;
var fixation2;
var letterpic2_1;
var letterpic2_2;
var check_instrClock;
var check;
var Learning_CheckClock;
var quest;
var letter_string_pic;
var number_string_pic;
var test_overClock;
var t_outro;
var task_overClock;
var task_outro;
var globalClock;
var routineTimer;
function experimentInit() {
  // Initialize components for Routine "Welcome"
  WelcomeClock = new util.Clock();
  text = new visual.TextStim({
    win: psychoJS.window,
    name: 'text',
    text: '\nPlease follow the instructions on the following screens.\n\n\nTo select a symbol on a given side of the screen, use the keys below:\n\nLeft  side =  C          Right side = M\n\n\nPress any key to continue. \n',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Initialize components for Routine "Prep_Instruction"
  Prep_InstructionClock = new util.Clock();
  instr1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'instr1',
    text: 'Short Preparation session. \n\nYou will be collecting points.\n\nChoose one symbol either on the left (C) or the right (M). \n\nOne symbol is more likely to give you 10 points.\nThe other symbol is more likely to take away 10 points. \n\nChoose the symbol that is the most likely to give you points and avoid the symbol that makes you lose points.\n\n\nPress any key to continue.\n',
    font: 'Arial',
    units : undefined, 
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Initialize components for Routine "prep"
  prepClock = new util.Clock();
  fixation0 = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixation0',
    text: '+',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.1,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  letterpic0_1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'letterpic0_1', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [(- 0.1), 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  letterpic0_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'letterpic0_2', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0.1, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "prep_feedback"
  prep_feedbackClock = new util.Clock();
  prep_feedb = new visual.TextStim({
    win: psychoJS.window,
    name: 'prep_feedb',
    text: 'default text',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.08,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: -1.0 
  });
  
  // Initialize components for Routine "Prep_over"
  Prep_overClock = new util.Clock();
  prep_outro = new visual.TextStim({
    win: psychoJS.window,
    name: 'prep_outro',
    text: 'The practice session is now finished.\n\nYou may have noticed that one of the symbols was more likely to give you points. \n\nThis was not always the case. \n\nIt was more PROBABLE that one symbol gave you points. \n\n\nPress any key to continue. ',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Initialize components for Routine "Practice_Instruction"
  Practice_InstructionClock = new util.Clock();
  myCount = 0
  
  
  pract_intr = new visual.TextStim({
    win: psychoJS.window,
    name: 'pract_intr',
    text: 'Training Session\n\nYou will see 3 pairs of symbols. \n\nIn each pair, one symbol is more PROBABLE to give you points. \n\nIdentify those symbols and choose them.\nAvoid symbols that make you lose points. \n\nUse buttons:\nleft(C)            right(M)\nto make your choices. \n\n\nPress any key to begin.\n',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  myCount1 = 0
  myCount2 = 0
  myCount3 = 0
  
  resp1 = []
  resp2 = []
  resp3 = []
  
  resplist1 = []
  resplist2 = []
  resplist3 = []
  
  Corr1 = 0
  Corr2 = 0
  Corr3 = 0
  fixation1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixation1',
    text: '+',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.1,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  letterpic1_1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'letterpic1_1', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [(- 0.1), 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  letterpic1_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'letterpic1_2', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0.1, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -3.0 
  });
  // Initialize components for Routine "feedback"
  feedbackClock = new util.Clock();
  blank_feed = new visual.TextStim({
    win: psychoJS.window,
    name: 'blank_feed',
    text: '',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.1,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  feedb = new visual.TextStim({
    win: psychoJS.window,
    name: 'feedb',
    text: 'default text',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.08,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: -2.0 
  });
  
  // Initialize components for Routine "Practice_over"
  Practice_overClock = new util.Clock();
  instr2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'instr2',
    text: 'The training session is now finished. \n\nPress any key to continue onto the next section.\n',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Initialize components for Routine "Test_Instr"
  Test_InstrClock = new util.Clock();
  instr3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'instr3',
    text: 'Test Session\n\nYou will see the same symbols again. \n\nNow, in different order and combination. \n\nPick the symbol that is the most PROBABLE to give you points.\nAvoid the symbol that makes you lose points.\n\nUse buttons:\nleft(C)            right(M)\n\nMake your decision as quickly as you can.\nIf you are unsure go with your gut feeling.\n\n\nPress any key to begin.\n',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Initialize components for Routine "test_trial"
  test_trialClock = new util.Clock();
  fixation2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixation2',
    text: '+',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.1,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  letterpic2_1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'letterpic2_1', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [(- 0.1), 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  letterpic2_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'letterpic2_2', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0.1, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -3.0 
  });
  // Initialize components for Routine "check_instr"
  check_instrClock = new util.Clock();
  check = new visual.TextStim({
    win: psychoJS.window,
    name: 'check',
    text: 'Well done!\n\nNow think about the symbols you saw. \n\nWhich symbol was the MOST likely to give you ponts? \n\nWhich symbol was the LEAST likely to give you points? \n\nOn the next screen, use numbers:\n1  2  3  4  5  or  6 on top of the keyboard.\n\nIdentify the symbols of your choice. \n\n\nPress any key to continue.',
    font: 'Arial',
    units : undefined, 
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Initialize components for Routine "Learning_Check"
  Learning_CheckClock = new util.Clock();
  quest = new visual.TextStim({
    win: psychoJS.window,
    name: 'quest',
    text: 'default text',
    font: 'Arial',
    units : undefined, 
    pos: [0, 0.3], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  letter_string_pic = new visual.ImageStim({
    win : psychoJS.window,
    name : 'letter_string_pic', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0.1], size : [0.8, 0.12],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  number_string_pic = new visual.ImageStim({
    win : psychoJS.window,
    name : 'number_string_pic', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, (- 0.1)], size : [0.8, 0.1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "test_over"
  test_overClock = new util.Clock();
  t_outro = new visual.TextStim({
    win: psychoJS.window,
    name: 't_outro',
    text: 'The test phase is now finished. \nYou can take a break before continuing. \n\nOtherwise press any key to continue.',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Initialize components for Routine "task_over"
  task_overClock = new util.Clock();
  task_outro = new visual.TextStim({
    win: psychoJS.window,
    name: 'task_outro',
    text: 'Thank you, the learning task is now finished.\n\nYou can press Esc to exit the full screen. \n\nYou can then close this window. ',
    font: 'Calibri',
    units : undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}

var t;
var frameN;
var welc_resp;
var WelcomeComponents;
function WelcomeRoutineBegin() {
  //------Prepare to start Routine 'Welcome'-------
  t = 0;
  WelcomeClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  welc_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  WelcomeComponents = [];
  WelcomeComponents.push(text);
  WelcomeComponents.push(welc_resp);
  
  WelcomeComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}

var continueRoutine;
function WelcomeRoutineEachFrame() {
  //------Loop for each frame of Routine 'Welcome'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = WelcomeClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *text* updates
  if (t >= 0.0 && text.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    text.tStart = t;  // (not accounting for frame time here)
    text.frameNStart = frameN;  // exact frame index
    text.setAutoDraw(true);
  }

  
  // *welc_resp* updates
  if (t >= 0.0 && welc_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    welc_resp.tStart = t;  // (not accounting for frame time here)
    welc_resp.frameNStart = frameN;  // exact frame index
    welc_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { welc_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (welc_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys();
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      welc_resp.keys = theseKeys[theseKeys.length-1];  // just the last key pressed
      welc_resp.rt = welc_resp.clock.getTime();
      // a response ends the routine
      continueRoutine = false;
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  WelcomeComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function WelcomeRoutineEnd() {
  //------Ending Routine 'Welcome'-------
  WelcomeComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (welc_resp.keys === undefined || welc_resp.keys.length === 0) {    // No response was made
      welc_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('welc_resp.keys', welc_resp.keys);
  if (typeof welc_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('welc_resp.rt', welc_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var intr0_resp;
var Prep_InstructionComponents;
function Prep_InstructionRoutineBegin() {
  //------Prepare to start Routine 'Prep_Instruction'-------
  t = 0;
  Prep_InstructionClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  intr0_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  Prep_InstructionComponents = [];
  Prep_InstructionComponents.push(instr1);
  Prep_InstructionComponents.push(intr0_resp);
  
  Prep_InstructionComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function Prep_InstructionRoutineEachFrame() {
  //------Loop for each frame of Routine 'Prep_Instruction'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = Prep_InstructionClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *instr1* updates
  if (t >= 0.0 && instr1.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    instr1.tStart = t;  // (not accounting for frame time here)
    instr1.frameNStart = frameN;  // exact frame index
    instr1.setAutoDraw(true);
  }

  
  // *intr0_resp* updates
  if (t >= 0.0 && intr0_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    intr0_resp.tStart = t;  // (not accounting for frame time here)
    intr0_resp.frameNStart = frameN;  // exact frame index
    intr0_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { intr0_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (intr0_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys();
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      intr0_resp.keys = theseKeys[theseKeys.length-1];  // just the last key pressed
      intr0_resp.rt = intr0_resp.clock.getTime();
      // a response ends the routine
      continueRoutine = false;
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  Prep_InstructionComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function Prep_InstructionRoutineEnd() {
  //------Ending Routine 'Prep_Instruction'-------
  Prep_InstructionComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (intr0_resp.keys === undefined || intr0_resp.keys.length === 0) {    // No response was made
      intr0_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('intr0_resp.keys', intr0_resp.keys);
  if (typeof intr0_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('intr0_resp.rt', intr0_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "Prep_Instruction" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var preparation;
var currentLoop;
var trialIterator;
function preparationLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  preparation = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'hiragana7.xlsx',
    seed: undefined, name: 'preparation'});
  psychoJS.experiment.addLoop(preparation); // add the loop to the experiment
  currentLoop = preparation;  // we're now the current loop

  // Schedule all the trials in the trialList:
  trialIterator = preparation[Symbol.iterator]();
  while(true) {
    let result = trialIterator.next();
    if (result.done);
      break;
    let thisPreparation = result.value;
    thisScheduler.add(importConditions(preparation));
    thisScheduler.add(prepRoutineBegin);
    thisScheduler.add(prepRoutineEachFrame);
    thisScheduler.add(prepRoutineEnd);
    thisScheduler.add(prep_feedbackRoutineBegin);
    thisScheduler.add(prep_feedbackRoutineEachFrame);
    thisScheduler.add(prep_feedbackRoutineEnd);
    thisScheduler.add(endLoopIteration(thisScheduler, thisPreparation));
  }

  return Scheduler.Event.NEXT;
}


function preparationLoopEnd() {
  psychoJS.experiment.removeLoop(preparation);

  return Scheduler.Event.NEXT;
}

var letterMaster;
function letterMasterLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  letterMaster = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'letterchoice.xlsx',
    seed: undefined, name: 'letterMaster'});
  psychoJS.experiment.addLoop(letterMaster); // add the loop to the experiment
  currentLoop = letterMaster;  // we're now the current loop

  // Schedule all the trials in the trialList:
  trialIterator = letterMaster[Symbol.iterator]();
  while(true) {
    let result = trialIterator.next();
    if (result.done);
      break;
    let thisLetterMaster = result.value;
    thisScheduler.add(importConditions(letterMaster));
    thisScheduler.add(Practice_InstructionRoutineBegin);
    thisScheduler.add(Practice_InstructionRoutineEachFrame);
    thisScheduler.add(Practice_InstructionRoutineEnd);
    const practiceLoopScheduler = new Scheduler(psychoJS);
    thisScheduler.add(practiceLoopBegin, practiceLoopScheduler);
    thisScheduler.add(practiceLoopScheduler);
    thisScheduler.add(practiceLoopEnd);
    thisScheduler.add(Practice_overRoutineBegin);
    thisScheduler.add(Practice_overRoutineEachFrame);
    thisScheduler.add(Practice_overRoutineEnd);
    thisScheduler.add(Test_InstrRoutineBegin);
    thisScheduler.add(Test_InstrRoutineEachFrame);
    thisScheduler.add(Test_InstrRoutineEnd);
    const testLoopScheduler = new Scheduler(psychoJS);
    thisScheduler.add(testLoopBegin, testLoopScheduler);
    thisScheduler.add(testLoopScheduler);
    thisScheduler.add(testLoopEnd);
    thisScheduler.add(check_instrRoutineBegin);
    thisScheduler.add(check_instrRoutineEachFrame);
    thisScheduler.add(check_instrRoutineEnd);
    const checksLoopScheduler = new Scheduler(psychoJS);
    thisScheduler.add(checksLoopBegin, checksLoopScheduler);
    thisScheduler.add(checksLoopScheduler);
    thisScheduler.add(checksLoopEnd);
    thisScheduler.add(test_overRoutineBegin);
    thisScheduler.add(test_overRoutineEachFrame);
    thisScheduler.add(test_overRoutineEnd);
    thisScheduler.add(endLoopIteration(thisScheduler, thisLetterMaster));
  }

  return Scheduler.Event.NEXT;
}

var practice;
function practiceLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  practice = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 6, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: letterset,
    seed: undefined, name: 'practice'});
  psychoJS.experiment.addLoop(practice); // add the loop to the experiment
  currentLoop = practice;  // we're now the current loop

  // Schedule all the trials in the trialList:
  trialIterator = practice[Symbol.iterator]();
  while(true) {
    let result = trialIterator.next();
    if (result.done);
      break;
    let thisPractice = result.value;
    thisScheduler.add(importConditions(practice));
    thisScheduler.add(trialRoutineBegin);
    thisScheduler.add(trialRoutineEachFrame);
    thisScheduler.add(trialRoutineEnd);
    thisScheduler.add(feedbackRoutineBegin);
    thisScheduler.add(feedbackRoutineEachFrame);
    thisScheduler.add(feedbackRoutineEnd);
    thisScheduler.add(endLoopIteration(thisScheduler, thisPractice));
  }

  return Scheduler.Event.NEXT;
}


function practiceLoopEnd() {
  psychoJS.experiment.removeLoop(practice);

  return Scheduler.Event.NEXT;
}

var test;
function testLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  test = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 2, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: letterset2,
    seed: undefined, name: 'test'});
  psychoJS.experiment.addLoop(test); // add the loop to the experiment
  currentLoop = test;  // we're now the current loop

  // Schedule all the trials in the trialList:
  trialIterator = test[Symbol.iterator]();
  while(true) {
    let result = trialIterator.next();
    if (result.done);
      break;
    let thisTest = result.value;
    thisScheduler.add(importConditions(test));
    thisScheduler.add(test_trialRoutineBegin);
    thisScheduler.add(test_trialRoutineEachFrame);
    thisScheduler.add(test_trialRoutineEnd);
    thisScheduler.add(endLoopIteration(thisScheduler, thisTest));
  }

  return Scheduler.Event.NEXT;
}


function testLoopEnd() {
  psychoJS.experiment.removeLoop(test);

  return Scheduler.Event.NEXT;
}

var checks;
function checksLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  checks = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
    extraInfo: expInfo, originPath: undefined,
    trialList: letterset3,
    seed: undefined, name: 'checks'});
  psychoJS.experiment.addLoop(checks); // add the loop to the experiment
  currentLoop = checks;  // we're now the current loop

  // Schedule all the trials in the trialList:
  trialIterator = checks[Symbol.iterator]();
  while(true) {
    let result = trialIterator.next();
    if (result.done);
      break;
    let thisCheck = result.value;
    thisScheduler.add(importConditions(checks));
    thisScheduler.add(Learning_CheckRoutineBegin);
    thisScheduler.add(Learning_CheckRoutineEachFrame);
    thisScheduler.add(Learning_CheckRoutineEnd);
    thisScheduler.add(endLoopIteration(thisScheduler, thisCheck));
  }

  return Scheduler.Event.NEXT;
}


function checksLoopEnd() {
  psychoJS.experiment.removeLoop(checks);

  return Scheduler.Event.NEXT;
}


function letterMasterLoopEnd() {
  psychoJS.experiment.removeLoop(letterMaster);

  return Scheduler.Event.NEXT;
}

var prep_resp;
var prepComponents;
function prepRoutineBegin() {
  //------Prepare to start Routine 'prep'-------
  t = 0;
  prepClock.reset(); // clock
  frameN = -1;
  routineTimer.add(2.000000);
  // update component parameters for each repeat
  letterpic0_1.setImage(letter10);
  letterpic0_2.setImage(letter20);
  prep_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  prepComponents = [];
  prepComponents.push(fixation0);
  prepComponents.push(letterpic0_1);
  prepComponents.push(letterpic0_2);
  prepComponents.push(prep_resp);
  
  prepComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}

var frameRemains;
function prepRoutineEachFrame() {
  //------Loop for each frame of Routine 'prep'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = prepClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *fixation0* updates
  if (t >= 0.0 && fixation0.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    fixation0.tStart = t;  // (not accounting for frame time here)
    fixation0.frameNStart = frameN;  // exact frame index
    fixation0.setAutoDraw(true);
  }

  frameRemains = 0.0 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (fixation0.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    fixation0.setAutoDraw(false);
  }
  
  // *letterpic0_1* updates
  if (t >= 0.5 && letterpic0_1.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    letterpic0_1.tStart = t;  // (not accounting for frame time here)
    letterpic0_1.frameNStart = frameN;  // exact frame index
    letterpic0_1.setAutoDraw(true);
  }

  frameRemains = 0.5 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (letterpic0_1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    letterpic0_1.setAutoDraw(false);
  }
  
  // *letterpic0_2* updates
  if (t >= 0.5 && letterpic0_2.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    letterpic0_2.tStart = t;  // (not accounting for frame time here)
    letterpic0_2.frameNStart = frameN;  // exact frame index
    letterpic0_2.setAutoDraw(true);
  }

  frameRemains = 0.5 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (letterpic0_2.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    letterpic0_2.setAutoDraw(false);
  }
  
  // *prep_resp* updates
  if (t >= 0.5 && prep_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    prep_resp.tStart = t;  // (not accounting for frame time here)
    prep_resp.frameNStart = frameN;  // exact frame index
    prep_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { prep_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  frameRemains = 0.5 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (prep_resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    prep_resp.status = PsychoJS.Status.FINISHED;
  }

  if (prep_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys({keyList:['c', 'm']});
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      if (prep_resp.keys.length === 0) {  // then this was the first keypress
        prep_resp.keys = theseKeys[0];  // just the first key pressed
        prep_resp.rt = prep_resp.clock.getTime();
        // was this 'correct'?
        if (prep_resp.keys == corrAns0) {
            prep_resp.corr = 1;
        } else {
            prep_resp.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  prepComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine && routineTimer.getTime() > 0) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function prepRoutineEnd() {
  //------Ending Routine 'prep'-------
  prepComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (prep_resp.keys === undefined || prep_resp.keys.length === 0) {    // No response was made
      prep_resp.keys = undefined;
  }
  
  // was no response the correct answer?!
  if (prep_resp.keys === undefined) {
    if (['None','none',undefined].includes(corrAns0)) {
       prep_resp.corr = 1  // correct non-response
    } else {
       prep_resp.corr = 0  // failed to respond (incorrectly)
    }
  }
  // store data for thisExp (ExperimentHandler)
  psychoJS.experiment.addData('prep_resp.keys', prep_resp.keys);
  psychoJS.experiment.addData('prep_resp.corr', prep_resp.corr);
  if (typeof prep_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('prep_resp.rt', prep_resp.rt);
      routineTimer.reset();
      }
  
  return Scheduler.Event.NEXT;
}

var msg0;
var msgColor;
var prep_feedbackComponents;
function prep_feedbackRoutineBegin() {
  //------Prepare to start Routine 'prep_feedback'-------
  t = 0;
  prep_feedbackClock.reset(); // clock
  frameN = -1;
  routineTimer.add(0.600000);
  // update component parameters for each repeat
  if(prep_resp.corr) {
    msg0='+10'
    msgColor='green';
  } if(!prep_resp.corr) {
    msg0='- 10'
    msgColor='red';
  } if(!prep_resp.keys) {
    msg0='No response detected'
    msgColor='black';
  }
  prep_feedb.setColor(new util.Color(msgColor));
  prep_feedb.setText(msg0);
  // keep track of which components have finished
  prep_feedbackComponents = [];
  prep_feedbackComponents.push(prep_feedb);
  
  prep_feedbackComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function prep_feedbackRoutineEachFrame() {
  //------Loop for each frame of Routine 'prep_feedback'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = prep_feedbackClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *prep_feedb* updates
  if (t >= 0.0 && prep_feedb.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    prep_feedb.tStart = t;  // (not accounting for frame time here)
    prep_feedb.frameNStart = frameN;  // exact frame index
    prep_feedb.setAutoDraw(true);
  }

  frameRemains = 0.0 + 0.6 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (prep_feedb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    prep_feedb.setAutoDraw(false);
  }
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  prep_feedbackComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine && routineTimer.getTime() > 0) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function prep_feedbackRoutineEnd() {
  //------Ending Routine 'prep_feedback'-------
  prep_feedbackComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  return Scheduler.Event.NEXT;
}

var outro_resp;
var Prep_overComponents;
function Prep_overRoutineBegin() {
  //------Prepare to start Routine 'Prep_over'-------
  t = 0;
  Prep_overClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  outro_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  Prep_overComponents = [];
  Prep_overComponents.push(prep_outro);
  Prep_overComponents.push(outro_resp);
  
  Prep_overComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function Prep_overRoutineEachFrame() {
  //------Loop for each frame of Routine 'Prep_over'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = Prep_overClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *prep_outro* updates
  if (t >= 0.0 && prep_outro.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    prep_outro.tStart = t;  // (not accounting for frame time here)
    prep_outro.frameNStart = frameN;  // exact frame index
    prep_outro.setAutoDraw(true);
  }

  
  // *outro_resp* updates
  if (t >= 0.0 && outro_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    outro_resp.tStart = t;  // (not accounting for frame time here)
    outro_resp.frameNStart = frameN;  // exact frame index
    outro_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { outro_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (outro_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys();
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      outro_resp.keys = theseKeys[theseKeys.length-1];  // just the last key pressed
      outro_resp.rt = outro_resp.clock.getTime();
      // a response ends the routine
      continueRoutine = false;
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  Prep_overComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function Prep_overRoutineEnd() {
  //------Ending Routine 'Prep_over'-------
  Prep_overComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (outro_resp.keys === undefined || outro_resp.keys.length === 0) {    // No response was made
      outro_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('outro_resp.keys', outro_resp.keys);
  if (typeof outro_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('outro_resp.rt', outro_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "Prep_over" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var instr1_resp;
var Practice_InstructionComponents;
function Practice_InstructionRoutineBegin() {
  //------Prepare to start Routine 'Practice_Instruction'-------
  t = 0;
  Practice_InstructionClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  myCount = myCount + 1
  
  if (myCount === 1){
      letterMaster.finished = true;
  }
  
  
  instr1_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  Practice_InstructionComponents = [];
  Practice_InstructionComponents.push(pract_intr);
  Practice_InstructionComponents.push(instr1_resp);
  
  Practice_InstructionComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function Practice_InstructionRoutineEachFrame() {
  //------Loop for each frame of Routine 'Practice_Instruction'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = Practice_InstructionClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *pract_intr* updates
  if (t >= 0.0 && pract_intr.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    pract_intr.tStart = t;  // (not accounting for frame time here)
    pract_intr.frameNStart = frameN;  // exact frame index
    pract_intr.setAutoDraw(true);
  }

  
  // *instr1_resp* updates
  if (t >= 0.0 && instr1_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    instr1_resp.tStart = t;  // (not accounting for frame time here)
    instr1_resp.frameNStart = frameN;  // exact frame index
    instr1_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { instr1_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (instr1_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys();
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      if (instr1_resp.keys.length === 0) {  // then this was the first keypress
        instr1_resp.keys = theseKeys[0];  // just the first key pressed
        instr1_resp.rt = instr1_resp.clock.getTime();
        // a response ends the routine
        continueRoutine = false;
      }
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  Practice_InstructionComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function Practice_InstructionRoutineEnd() {
  //------Ending Routine 'Practice_Instruction'-------
  Practice_InstructionComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (instr1_resp.keys === undefined || instr1_resp.keys.length === 0) {    // No response was made
      instr1_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('instr1_resp.keys', instr1_resp.keys);
  if (typeof instr1_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('instr1_resp.rt', instr1_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "Practice_Instruction" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var practiceA;
var practiceB;
var practiceC;
var arrSum;
var jitter;
var jitter2;
var resp;
var trialComponents;
function trialRoutineBegin() {
  //------Prepare to start Routine 'trial'-------
  t = 0;
  trialClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  if(myCount == 1) {
    myCount1 = myCount1 + 1;
  } else {
    myCount1 = myCount1 + 0;
  }
  
  if(myCount == 2) {
    myCount2 = myCount2 + 1;
  } else {
    myCount2 = myCount2 + 0;
  }
  
  if(myCount == 3) {
    myCount3 = myCount3 + 1;
  } else {
    myCount3 = myCount3 + 0;
  }
  
  
  practiceA = resplist1.slice(-60)
  practiceB = resplist2.slice(-60)
  practiceC = resplist3.slice(-60)
  
  
  arrSum = function(arr){
    return arr.reduce(function(a,b){
      return a + b
    }, 0);
  }
  
  
  Corr1 = arrSum(practiceA)
  Corr2 = arrSum(practiceB)
  Corr3 = arrSum(practiceC)
  
  
  if (myCount == 1 && Corr1 > 13 && Corr2 > 11 && Corr3 > 9 && myCount1 > 59){
      practice.finished = true;
  }
  
  if (myCount == 2 && Corr1 > 13 && Corr2 > 11 && Corr3 > 9 && myCount2 > 59){
      practice.finished = true;
  } 
  
  if (myCount == 3 && Corr1 > 13 && Corr2 > 11 && Corr3 > 9 && myCount3 > 59){
      practice.finished = true;
  }
  
  
  jitter = Math.random() * (0.8 - 0.3) + 0.3;
   
  jitter = round(jitter, 1);
  
  
  jitter2 = Math.random() * (1.0 - 0.5) + 0.5;
   
  jitter2 = round(jitter2, 1);
  
  function round(value, precision) {
      var x = Math.pow(10, precision);
      return Math.round(value * x) / x;
  }
  
  
  letterpic1_1.setImage(letter1);
  letterpic1_2.setImage(letter2);
  resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  trialComponents = [];
  trialComponents.push(fixation1);
  trialComponents.push(letterpic1_1);
  trialComponents.push(letterpic1_2);
  trialComponents.push(resp);
  
  trialComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function trialRoutineEachFrame() {
  //------Loop for each frame of Routine 'trial'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = trialClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *fixation1* updates
  if (t >= 0.0 && fixation1.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    fixation1.tStart = t;  // (not accounting for frame time here)
    fixation1.frameNStart = frameN;  // exact frame index
    fixation1.setAutoDraw(true);
  }

  frameRemains = 0.0 + jitter - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (fixation1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    fixation1.setAutoDraw(false);
  }
  
  // *letterpic1_1* updates
  if (t >= jitter && letterpic1_1.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    letterpic1_1.tStart = t;  // (not accounting for frame time here)
    letterpic1_1.frameNStart = frameN;  // exact frame index
    letterpic1_1.setAutoDraw(true);
  }

  frameRemains = jitter + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (letterpic1_1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    letterpic1_1.setAutoDraw(false);
  }
  
  // *letterpic1_2* updates
  if (t >= jitter && letterpic1_2.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    letterpic1_2.tStart = t;  // (not accounting for frame time here)
    letterpic1_2.frameNStart = frameN;  // exact frame index
    letterpic1_2.setAutoDraw(true);
  }

  frameRemains = jitter + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (letterpic1_2.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    letterpic1_2.setAutoDraw(false);
  }
  
  // *resp* updates
  if (t >= jitter && resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    resp.tStart = t;  // (not accounting for frame time here)
    resp.frameNStart = frameN;  // exact frame index
    resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  frameRemains = jitter + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    resp.status = PsychoJS.Status.FINISHED;
  }

  if (resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys({keyList:['c', 'm']});
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      if (resp.keys.length === 0) {  // then this was the first keypress
        resp.keys = theseKeys[0];  // just the first key pressed
        resp.rt = resp.clock.getTime();
        // was this 'correct'?
        if (resp.keys == corrAns) {
            resp.corr = 1;
        } else {
            resp.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  trialComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function trialRoutineEnd() {
  //------Ending Routine 'trial'-------
  trialComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  if(resp.keys == letterA) {
    resp1 = 1;
  } else {
    resp1 = 0;
  }
  
  if(resp.keys == letterC) {
    resp2 = 1;
  } else {
    resp2 = 0;
  }
  
  if(resp.keys == letterE) {
    resp3 = 1;
  } else {
    resp3 = 0;
  }
  
  resplist1.push(resp1)
  resplist2.push(resp2)
  resplist3.push(resp3)
  
  
  
  // check responses
  if (resp.keys === undefined || resp.keys.length === 0) {    // No response was made
      resp.keys = undefined;
  }
  
  // was no response the correct answer?!
  if (resp.keys === undefined) {
    if (['None','none',undefined].includes(corrAns)) {
       resp.corr = 1  // correct non-response
    } else {
       resp.corr = 0  // failed to respond (incorrectly)
    }
  }
  // store data for thisExp (ExperimentHandler)
  psychoJS.experiment.addData('resp.keys', resp.keys);
  psychoJS.experiment.addData('resp.corr', resp.corr);
  if (typeof resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('resp.rt', resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "trial" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var msg;
var feedbackComponents;
function feedbackRoutineBegin() {
  //------Prepare to start Routine 'feedback'-------
  t = 0;
  feedbackClock.reset(); // clock
  frameN = -1;
  routineTimer.add(0.950000);
  // update component parameters for each repeat
  if(resp.corr) {
    msg='+10'
    msgColor='green';
  } if(!resp.corr) {
    msg='- 10'
    msgColor='red';
  } if(!resp.keys) {
    msg='No response detected'
    msgColor='black';
  }
  feedb.setColor(new util.Color(msgColor));
  feedb.setText(msg);
  // keep track of which components have finished
  feedbackComponents = [];
  feedbackComponents.push(blank_feed);
  feedbackComponents.push(feedb);
  
  feedbackComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function feedbackRoutineEachFrame() {
  //------Loop for each frame of Routine 'feedback'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = feedbackClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *blank_feed* updates
  if (t >= 0.0 && blank_feed.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    blank_feed.tStart = t;  // (not accounting for frame time here)
    blank_feed.frameNStart = frameN;  // exact frame index
    blank_feed.setAutoDraw(true);
  }

  frameRemains = 0.0 + 0.35 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (blank_feed.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    blank_feed.setAutoDraw(false);
  }
  
  // *feedb* updates
  if (t >= 0.35 && feedb.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    feedb.tStart = t;  // (not accounting for frame time here)
    feedb.frameNStart = frameN;  // exact frame index
    feedb.setAutoDraw(true);
  }

  frameRemains = 0.35 + 0.6 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (feedb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    feedb.setAutoDraw(false);
  }
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  feedbackComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine && routineTimer.getTime() > 0) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function feedbackRoutineEnd() {
  //------Ending Routine 'feedback'-------
  feedbackComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  return Scheduler.Event.NEXT;
}

var instr2_resp;
var Practice_overComponents;
function Practice_overRoutineBegin() {
  //------Prepare to start Routine 'Practice_over'-------
  t = 0;
  Practice_overClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  instr2_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  Practice_overComponents = [];
  Practice_overComponents.push(instr2);
  Practice_overComponents.push(instr2_resp);
  
  Practice_overComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function Practice_overRoutineEachFrame() {
  //------Loop for each frame of Routine 'Practice_over'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = Practice_overClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *instr2* updates
  if (t >= 0.0 && instr2.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    instr2.tStart = t;  // (not accounting for frame time here)
    instr2.frameNStart = frameN;  // exact frame index
    instr2.setAutoDraw(true);
  }

  
  // *instr2_resp* updates
  if (t >= 0.0 && instr2_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    instr2_resp.tStart = t;  // (not accounting for frame time here)
    instr2_resp.frameNStart = frameN;  // exact frame index
    instr2_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { instr2_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (instr2_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys();
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      if (instr2_resp.keys.length === 0) {  // then this was the first keypress
        instr2_resp.keys = theseKeys[0];  // just the first key pressed
        instr2_resp.rt = instr2_resp.clock.getTime();
        // a response ends the routine
        continueRoutine = false;
      }
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  Practice_overComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function Practice_overRoutineEnd() {
  //------Ending Routine 'Practice_over'-------
  Practice_overComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (instr2_resp.keys === undefined || instr2_resp.keys.length === 0) {    // No response was made
      instr2_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('instr2_resp.keys', instr2_resp.keys);
  if (typeof instr2_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('instr2_resp.rt', instr2_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "Practice_over" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var instr3_resp;
var Test_InstrComponents;
function Test_InstrRoutineBegin() {
  //------Prepare to start Routine 'Test_Instr'-------
  t = 0;
  Test_InstrClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  instr3_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  Test_InstrComponents = [];
  Test_InstrComponents.push(instr3);
  Test_InstrComponents.push(instr3_resp);
  
  Test_InstrComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function Test_InstrRoutineEachFrame() {
  //------Loop for each frame of Routine 'Test_Instr'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = Test_InstrClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *instr3* updates
  if (t >= 0.0 && instr3.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    instr3.tStart = t;  // (not accounting for frame time here)
    instr3.frameNStart = frameN;  // exact frame index
    instr3.setAutoDraw(true);
  }

  
  // *instr3_resp* updates
  if (t >= 0.0 && instr3_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    instr3_resp.tStart = t;  // (not accounting for frame time here)
    instr3_resp.frameNStart = frameN;  // exact frame index
    instr3_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { instr3_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (instr3_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys();
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      instr3_resp.keys = theseKeys[theseKeys.length-1];  // just the last key pressed
      instr3_resp.rt = instr3_resp.clock.getTime();
      // a response ends the routine
      continueRoutine = false;
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  Test_InstrComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function Test_InstrRoutineEnd() {
  //------Ending Routine 'Test_Instr'-------
  Test_InstrComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (instr3_resp.keys === undefined || instr3_resp.keys.length === 0) {    // No response was made
      instr3_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('instr3_resp.keys', instr3_resp.keys);
  if (typeof instr3_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('instr3_resp.rt', instr3_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "Test_Instr" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var test_resp;
var test_trialComponents;
function test_trialRoutineBegin() {
  //------Prepare to start Routine 'test_trial'-------
  t = 0;
  test_trialClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  jitter = Math.random() * (0.8 - 0.3) + 0.3;
   
  jitter = round(jitter, 1);
  
  
  jitter2 = Math.random() * (1.0 - 0.5) + 0.5;
   
  jitter2 = round(jitter2, 1);
  
  function round(value, precision) {
      var x = Math.pow(10, precision);
      return Math.round(value * x) / x;
  }
  
  
  letterpic2_1.setImage(letter3);
  letterpic2_2.setImage(letter4);
  test_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  test_trialComponents = [];
  test_trialComponents.push(fixation2);
  test_trialComponents.push(letterpic2_1);
  test_trialComponents.push(letterpic2_2);
  test_trialComponents.push(test_resp);
  
  test_trialComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function test_trialRoutineEachFrame() {
  //------Loop for each frame of Routine 'test_trial'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = test_trialClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *fixation2* updates
  if (t >= 0.0 && fixation2.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    fixation2.tStart = t;  // (not accounting for frame time here)
    fixation2.frameNStart = frameN;  // exact frame index
    fixation2.setAutoDraw(true);
  }

  frameRemains = 0.0 + jitter2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (fixation2.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    fixation2.setAutoDraw(false);
  }
  
  // *letterpic2_1* updates
  if (t >= jitter2 && letterpic2_1.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    letterpic2_1.tStart = t;  // (not accounting for frame time here)
    letterpic2_1.frameNStart = frameN;  // exact frame index
    letterpic2_1.setAutoDraw(true);
  }

  frameRemains = jitter2 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (letterpic2_1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    letterpic2_1.setAutoDraw(false);
  }
  
  // *letterpic2_2* updates
  if (t >= jitter2 && letterpic2_2.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    letterpic2_2.tStart = t;  // (not accounting for frame time here)
    letterpic2_2.frameNStart = frameN;  // exact frame index
    letterpic2_2.setAutoDraw(true);
  }

  frameRemains = jitter2 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (letterpic2_2.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    letterpic2_2.setAutoDraw(false);
  }
  
  // *test_resp* updates
  if (t >= jitter2 && test_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    test_resp.tStart = t;  // (not accounting for frame time here)
    test_resp.frameNStart = frameN;  // exact frame index
    test_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { test_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  frameRemains = jitter2 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
  if (test_resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
    test_resp.status = PsychoJS.Status.FINISHED;
  }

  if (test_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys({keyList:['c', 'm']});
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      if (test_resp.keys.length === 0) {  // then this was the first keypress
        test_resp.keys = theseKeys[0];  // just the first key pressed
        test_resp.rt = test_resp.clock.getTime();
        // a response ends the routine
        continueRoutine = false;
      }
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  test_trialComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function test_trialRoutineEnd() {
  //------Ending Routine 'test_trial'-------
  test_trialComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (test_resp.keys === undefined || test_resp.keys.length === 0) {    // No response was made
      test_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('test_resp.keys', test_resp.keys);
  if (typeof test_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('test_resp.rt', test_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "test_trial" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var check_resp;
var check_instrComponents;
function check_instrRoutineBegin() {
  //------Prepare to start Routine 'check_instr'-------
  t = 0;
  check_instrClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  check_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  check_instrComponents = [];
  check_instrComponents.push(check);
  check_instrComponents.push(check_resp);
  
  check_instrComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function check_instrRoutineEachFrame() {
  //------Loop for each frame of Routine 'check_instr'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = check_instrClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *check* updates
  if (t >= 0.0 && check.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    check.tStart = t;  // (not accounting for frame time here)
    check.frameNStart = frameN;  // exact frame index
    check.setAutoDraw(true);
  }

  
  // *check_resp* updates
  if (t >= 0.0 && check_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    check_resp.tStart = t;  // (not accounting for frame time here)
    check_resp.frameNStart = frameN;  // exact frame index
    check_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { check_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (check_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys();
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      check_resp.keys = theseKeys[theseKeys.length-1];  // just the last key pressed
      check_resp.rt = check_resp.clock.getTime();
      // a response ends the routine
      continueRoutine = false;
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  check_instrComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function check_instrRoutineEnd() {
  //------Ending Routine 'check_instr'-------
  check_instrComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (check_resp.keys === undefined || check_resp.keys.length === 0) {    // No response was made
      check_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('check_resp.keys', check_resp.keys);
  if (typeof check_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('check_resp.rt', check_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "check_instr" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var learnt_resp;
var Learning_CheckComponents;
function Learning_CheckRoutineBegin() {
  //------Prepare to start Routine 'Learning_Check'-------
  t = 0;
  Learning_CheckClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  quest.setText(question);
  letter_string_pic.setImage(learnt_letters);
  number_string_pic.setImage(numbers);
  learnt_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  Learning_CheckComponents = [];
  Learning_CheckComponents.push(quest);
  Learning_CheckComponents.push(letter_string_pic);
  Learning_CheckComponents.push(number_string_pic);
  Learning_CheckComponents.push(learnt_resp);
  
  Learning_CheckComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function Learning_CheckRoutineEachFrame() {
  //------Loop for each frame of Routine 'Learning_Check'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = Learning_CheckClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *quest* updates
  if (t >= 0.0 && quest.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    quest.tStart = t;  // (not accounting for frame time here)
    quest.frameNStart = frameN;  // exact frame index
    quest.setAutoDraw(true);
  }

  
  // *letter_string_pic* updates
  if (t >= 0.0 && letter_string_pic.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    letter_string_pic.tStart = t;  // (not accounting for frame time here)
    letter_string_pic.frameNStart = frameN;  // exact frame index
    letter_string_pic.setAutoDraw(true);
  }

  
  // *number_string_pic* updates
  if (t >= 0.0 && number_string_pic.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    number_string_pic.tStart = t;  // (not accounting for frame time here)
    number_string_pic.frameNStart = frameN;  // exact frame index
    number_string_pic.setAutoDraw(true);
  }

  
  // *learnt_resp* updates
  if (t >= 0.0 && learnt_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    learnt_resp.tStart = t;  // (not accounting for frame time here)
    learnt_resp.frameNStart = frameN;  // exact frame index
    learnt_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { learnt_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (learnt_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys({keyList:['1', '2', '3', '4', '5', '6']});
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      learnt_resp.keys = theseKeys[theseKeys.length-1];  // just the last key pressed
      learnt_resp.rt = learnt_resp.clock.getTime();
      // a response ends the routine
      continueRoutine = false;
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  Learning_CheckComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function Learning_CheckRoutineEnd() {
  //------Ending Routine 'Learning_Check'-------
  Learning_CheckComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (learnt_resp.keys === undefined || learnt_resp.keys.length === 0) {    // No response was made
      learnt_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('learnt_resp.keys', learnt_resp.keys);
  if (typeof learnt_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('learnt_resp.rt', learnt_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "Learning_Check" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var outro_resp2;
var test_overComponents;
function test_overRoutineBegin() {
  //------Prepare to start Routine 'test_over'-------
  t = 0;
  test_overClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  outro_resp2 = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  test_overComponents = [];
  test_overComponents.push(t_outro);
  test_overComponents.push(outro_resp2);
  
  test_overComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function test_overRoutineEachFrame() {
  //------Loop for each frame of Routine 'test_over'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = test_overClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *t_outro* updates
  if (t >= 0.0 && t_outro.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    t_outro.tStart = t;  // (not accounting for frame time here)
    t_outro.frameNStart = frameN;  // exact frame index
    t_outro.setAutoDraw(true);
  }

  
  // *outro_resp2* updates
  if (t >= 0.0 && outro_resp2.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    outro_resp2.tStart = t;  // (not accounting for frame time here)
    outro_resp2.frameNStart = frameN;  // exact frame index
    outro_resp2.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { outro_resp2.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (outro_resp2.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys();
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      outro_resp2.keys = theseKeys[theseKeys.length-1];  // just the last key pressed
      outro_resp2.rt = outro_resp2.clock.getTime();
      // a response ends the routine
      continueRoutine = false;
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  test_overComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function test_overRoutineEnd() {
  //------Ending Routine 'test_over'-------
  test_overComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (outro_resp2.keys === undefined || outro_resp2.keys.length === 0) {    // No response was made
      outro_resp2.keys = undefined;
  }
  
  psychoJS.experiment.addData('outro_resp2.keys', outro_resp2.keys);
  if (typeof outro_resp2.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('outro_resp2.rt', outro_resp2.rt);
      routineTimer.reset();
      }
  
  // the Routine "test_over" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}

var outro2_resp;
var task_overComponents;
function task_overRoutineBegin() {
  //------Prepare to start Routine 'task_over'-------
  t = 0;
  task_overClock.reset(); // clock
  frameN = -1;
  // update component parameters for each repeat
  outro2_resp = new core.BuilderKeyResponse(psychoJS);
  
  // keep track of which components have finished
  task_overComponents = [];
  task_overComponents.push(task_outro);
  task_overComponents.push(outro2_resp);
  
  task_overComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent)
      thisComponent.status = PsychoJS.Status.NOT_STARTED;
     });
  
  return Scheduler.Event.NEXT;
}


function task_overRoutineEachFrame() {
  //------Loop for each frame of Routine 'task_over'-------
  let continueRoutine = true; // until we're told otherwise
  // get current time
  t = task_overClock.getTime();
  frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
  // update/draw components on each frame
  
  // *task_outro* updates
  if (t >= 0.0 && task_outro.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    task_outro.tStart = t;  // (not accounting for frame time here)
    task_outro.frameNStart = frameN;  // exact frame index
    task_outro.setAutoDraw(true);
  }

  
  // *outro2_resp* updates
  if (t >= 0.0 && outro2_resp.status === PsychoJS.Status.NOT_STARTED) {
    // keep track of start time/frame for later
    outro2_resp.tStart = t;  // (not accounting for frame time here)
    outro2_resp.frameNStart = frameN;  // exact frame index
    outro2_resp.status = PsychoJS.Status.STARTED;
    // keyboard checking is just starting
    psychoJS.window.callOnFlip(function() { outro2_resp.clock.reset(); }); // t = 0 on screen flip
    psychoJS.eventManager.clearEvents({eventType:'keyboard'});
  }

  if (outro2_resp.status === PsychoJS.Status.STARTED) {
    let theseKeys = psychoJS.eventManager.getKeys();
    
    // check for quit:
    if (theseKeys.indexOf('escape') > -1) {
      psychoJS.experiment.experimentEnded = true;
    }
    
    if (theseKeys.length > 0) {  // at least one key was pressed
      if (outro2_resp.keys.length === 0) {  // then this was the first keypress
        outro2_resp.keys = theseKeys[0];  // just the first key pressed
        outro2_resp.rt = outro2_resp.clock.getTime();
        // a response ends the routine
        continueRoutine = false;
      }
    }
  }
  
  // check for quit (typically the Esc key)
  if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
    return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
  }
  
  // check if the Routine should terminate
  if (!continueRoutine) {  // a component has requested a forced-end of Routine
    return Scheduler.Event.NEXT;
  }
  
  continueRoutine = false;  // reverts to True if at least one component still running
  task_overComponents.forEach( function(thisComponent) {
    if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
      continueRoutine = true;
    }});
  
  // refresh the screen if continuing
  if (continueRoutine) {
    return Scheduler.Event.FLIP_REPEAT;
  }
  else {
    return Scheduler.Event.NEXT;
  }
}


function task_overRoutineEnd() {
  //------Ending Routine 'task_over'-------
  task_overComponents.forEach( function(thisComponent) {
    if (typeof thisComponent.setAutoDraw === 'function') {
      thisComponent.setAutoDraw(false);
    }});
  
  // check responses
  if (outro2_resp.keys === undefined || outro2_resp.keys.length === 0) {    // No response was made
      outro2_resp.keys = undefined;
  }
  
  psychoJS.experiment.addData('outro2_resp.keys', outro2_resp.keys);
  if (typeof outro2_resp.keys !== 'undefined') {  // we had a response
      psychoJS.experiment.addData('outro2_resp.rt', outro2_resp.rt);
      routineTimer.reset();
      }
  
  // the Routine "task_over" was not non-slip safe, so reset the non-slip timer
  routineTimer.reset();
  
  return Scheduler.Event.NEXT;
}


function endLoopIteration(thisScheduler, thisTrial) {
  // ------Prepare for next entry------
  return function () {
    // ------Check if user ended loop early------
    if (currentLoop.finished) {
      thisScheduler.stop();
    } else if (typeof thisTrial === 'undefined' || !('isTrials' in thisTrial) || thisTrial.isTrials) {
      psychoJS.experiment.nextEntry();
    }
  return Scheduler.Event.NEXT;
  };
}


function importConditions(loop) {
  const trialIndex = loop.getTrialIndex();
  return function () {
    loop.setTrialIndex(trialIndex);
    psychoJS.importAttributes(loop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


function quitPsychoJS(message, isCompleted) {
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});

  return Scheduler.Event.QUIT;
}
