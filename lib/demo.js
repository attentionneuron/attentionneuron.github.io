// demo

var cartpole_demo = function(settings) {
  "use strict";

  var divName = settings.divName;

  var cartpole_sketch = function( p ) { 
    "use strict";

    var screen_width, screen_height; // stores the browser's dimensions
    var actual_screen_width, actual_screen_height;
    var full_screen_width, full_screen_height;
    var screen_y; // window.innerHeight

    var origx, origy;

    function shuffle(array) {
      var currentIndex = array.length, temporaryValue, randomIndex;

      // While there remain elements to shuffle...
      while (0 !== currentIndex) {

        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;

        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
      }

      return array;
    }

    // UI
    var canvas;
    var idx = [0, 1, 2, 3, 4];
    idx = shuffle(idx);
    var title_text="Permutation Order: ["+idx+"]";
    var shuffle_button, restart_button;
    var shuffle_button_mode = true;

    var input_labels = ["x", "ẋ", "cos(θ)", "sin(θ)", "θ̇"];
    var augmented_input_labels = new Array(5);
    var obs_history;

    for (var i=0;i<5;i++) {
      augmented_input_labels[idx[i]] = input_labels[i];
    }

    // agent
    var env = new CartPoleSwingUpEnv( p );

    var obs, reward, done, result, action, prev_action;
    var augmented_obs, hidden_state;
    var model;

    var set_screen = function() {
      // make sure we enforce some minimum size of our demo
      actual_screen_width = Math.max(window.innerWidth, 320);
      actual_screen_height = Math.max(window.innerHeight, 320);
      // make a 4x1 resolution demo.
      var screen_dim = actual_screen_width;
      screen_width = screen_dim;
      screen_height = screen_dim / 4.0;
    }

    function active_screen() {
      var result = false;
      var app_top = origy;
      var app_bottom = origy+screen_y;

      if ((app_top >= scroll_top) && (app_top <= (scroll_top+full_screen_height))) {
        result = true;
      }
      if ((app_bottom >= scroll_top) && (app_bottom <= (scroll_top+full_screen_height))) {
        result = true;
      }
      /* debug
      if (!result) {
        console.log("app_top", app_top, "app_bottom", app_bottom, "scroll_top", scroll_top, "full_screen_height", full_screen_height);        
      }
      */
     return result;
    }

    var set_screen = function() {
      // make sure we enforce some minimum size of our demo
      actual_screen_width = Math.max(window.document.getElementById(divName).parentElement.clientWidth, 320);
      actual_screen_height = Math.max(window.document.getElementById(divName).parentElement.clientHeight, 320);
      full_screen_height = window.innerHeight;
      full_screen_width = window.innerWidth;

      screen_y = window.innerHeight;

      var bodyRect = document.body.getBoundingClientRect()
      var rect = window.document.getElementById(divName).getBoundingClientRect();
      origy = rect.top - bodyRect.top;
      origx = rect.left - bodyRect.left;
      // make a 4x1 resolution demo.
      var screen_dim = actual_screen_width;
      screen_width = screen_dim;
      screen_height = screen_dim / 4.0;
    }

    var restart = function() {

      set_screen();

      // reset env
      obs = env.reset();

      // model setup
      prev_action = 0;
      augmented_obs = nj.zeros([5, 2]);
      model = new CartpoleSwingupModel();
      hidden_state = model.zero_state();
      obs_history = [[], [], [], [], []];

      canvas = p.createCanvas(screen_width, screen_height+60+215);
      p.frameRate(60);
      p.background(255, 255, 255, 255);
      p.fill(255, 255, 255, 255);

      var font_name = "Courier"
      p.textFont(font_name);
      p.textSize(16);

      var canvas_position = canvas.position();
      var cx = canvas_position.x;
      var cy = canvas_position.y;

      restart_button = p.createButton('restart environment');
      restart_button.style("font-family", font_name);
      restart_button.style("font-size", "16");

      restart_button.position(cx+screen_width*0.02, cy+screen_height+35+25);
      restart_button.mousePressed(restart_environment);


      shuffle_button = p.createButton('shuffle observations');
      shuffle_button.style("font-family", font_name);
      shuffle_button.style("font-size", "16");

      shuffle_button.position(cx+screen_width*0.04+restart_button.size().width, cy+screen_height+35+25);
      shuffle_button.mousePressed(shuffle_observation_order);


    };

    var reset_screen = function() {
      set_screen();
      p.resizeCanvas(screen_width, screen_height+60+215);
      var canvas_position = canvas.position();
      var cx = canvas_position.x;
      var cy = canvas_position.y;
      restart_button.position(cx+screen_width*0.02, cy+screen_height+35+25);
      restart_button.mousePressed(restart_environment);

      shuffle_button.position(cx+screen_width*0.04+restart_button.size().width, cy+screen_height+35+25);
      shuffle_button.mousePressed(shuffle_observation_order);

    }

    p.windowResized = function() {
      reset_screen();
    }

    var shuffle_observation_order = function() {
      idx = shuffle(idx);
      title_text="Permutation Order: ["+idx+"]";

      for (var i=0;i<5;i++) {
        augmented_input_labels[idx[i]] = input_labels[i];
      }
    }

    var restart_environment = function() {
      obs = env.reset();
      obs_history = [[], [], [], [], []];
    }

    p.setup = function() {
      restart(); // initialize variables for this demo and redraws interface
    };

    p.draw = function() {

      if (active_screen()) {

        // clear screen
        p.background(255);

        env.render(screen_width);

        for (var i=0;i<5;i++) {
          augmented_obs.set(idx[i], 0, obs[i]);
          augmented_obs.set(i, 1, prev_action);
          var o = obs[i];
          if (i==1 || i == 4) {
            o = Math.tanh(obs[i]); // keep stuff between -1 and 1
          }
          if (i == 0) {
            o = obs[i] / env.x_threshold;
          }
          obs_history[idx[i]].push(o);
        }
        [action, hidden_state] = model.forward(augmented_obs, hidden_state);

        // gym loop
        result = env.step(action);
        prev_action = action;

        obs = result[0];
        reward = result[1];
        done = result[2];

        if (done) {
          obs = env.reset();
          obs_history = [[], [], [], [], []];
        }

        // historical information.
        var history_area = screen_width/4+60+15+50;
        var history_length = obs_history[0].length;
        var delta_x = (screen_width-170-40) / 120;
        var start_x = screen_width*0.02+170;
        p.stroke(0);
        p.strokeWeight(0.5);
        for(var i=0;i<5;i++) {
          p.text("Input "+i+": "+augmented_input_labels[i], screen_width*0.02, history_area+30*i)
          //p.text(obs_history[i][history_length-1], screen_width*0.02+150, history_area+30*i)
          var o = obs_history[i];
          var y = 0;
          for(var j=0;j<120;j++) {
            if (j > history_length-1) {
              break;
            }

              y = o[history_length-1-j];

            p.line(start_x+delta_x*j, history_area+30*i+-7, start_x+delta_x*j, history_area+30*i-7+10*y);
          }
        }

        // draw text
        p.text(title_text, screen_width*0.02, history_area-82.5);

      }

    };

  };

  return cartpole_sketch;

};

