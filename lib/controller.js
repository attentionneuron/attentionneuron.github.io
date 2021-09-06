// controls the flow of the page

var scroll_top = 0;
var scroll_bottom = 1;
$(document).scroll(function(){
  scroll_top = $(document).scrollTop();
  scroll_bottom = scroll_top + $(document).height();
});

// blazy code
var bLazy = new Blazy({
  success: function(){
    updateCounter();
  }
});

// not needed, only here to illustrate amount of loaded images
var imageLoaded = 0;

function updateCounter() {
  imageLoaded++;
  console.log("blazy image loaded: "+imageLoaded);
}

var intro_demo_settings = {
  divName: 'intro_demo',
};
var intro_demo_instance = new p5(cartpole_demo(intro_demo_settings), 'intro_demo');

var cartpole_demo_settings = {
  divName: 'cartpole_demo',
};
var cartpole_demo_instance = new p5(cartpole_demo(cartpole_demo_settings), 'cartpole_demo');

var cartpole_demo_special_settings = {
  divName: 'cartpole_demo_special',
};
var cartpole_demo_special_instance = new p5(cartpole_demo_special(cartpole_demo_special_settings), 'cartpole_demo_special');
