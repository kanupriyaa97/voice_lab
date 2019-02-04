/* Copyright 2013 Chris Wilson

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   */

   window.AudioContext = window.AudioContext || window.webkitAudioContext;
   navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia;
   window.URL = window.URL || window.webkitURL;
   navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;


   var audioContext = new AudioContext();
   
   var audioInput = null,
   realAudioInput = null,
   inputPoint = null,
   audioRecorder = null;
   var rafID = null;
   var analyserContext = null;
   var canvasWidth, canvasHeight;
   var recIndex = 0;

   var g = null;
   var g2 = null;

   var spectro;
   var microphoneButton;
   var songButton;
   var songSelect;
   var selectedMedia;

   function gotBuffers( buffers ) {
    var canvas = document.getElementById( "wavedisplay" );
    //drawBuffer( canvas.width, canvas.height, canvas.getContext('2d'), buffers[0] );

    // the ONLY time gotBuffers is called is right after a new recording is completed - 
    // so here's where we should set up the download.
    audioRecorder.exportMonoWAV( doneEncoding );

  }

  function doneEncoding( blob ) {
    var link = document.getElementById("save");
    link.className = "active";

    var fd = new FormData();
    fd.append('file', blob);
    $.ajax({
      type: 'POST',
      url: '/upload',
      data: fd,
      processData: false,
      contentType: false
    })

  }
  var INTERVALID = 0
  function toggleRecording( e ) {
    if (e.classList.contains("recording")) {
        // stop recording
        
        e.classList.remove("recording");
        clearInterval(INTERVALID)
      } else {
        // start recording
        if (!audioRecorder)
          return;
        e.classList.add("recording");
        var recording = false;
        
        INTERVALID = setInterval(function(){
          if (recording){
            audioRecorder.stop();
            audioRecorder.getBuffers( function(buffer){
              gotBuffers(buffer)
              audioRecorder.clear();
              audioRecorder.record();
            });
          }
          else{
            audioRecorder.clear();
            audioRecorder.record();
          }
          recording = true;
        }, 7000)
        
      }
    }

    function convertToMono( input ) {
      var splitter = audioContext.createChannelSplitter(2);
      var merger = audioContext.createChannelMerger(2);

      input.connect( splitter );
      splitter.connect( merger, 0, 0 );
      splitter.connect( merger, 0, 1 );
      return merger;
    }

    function cancelAnalyserUpdates() {
      window.cancelAnimationFrame( rafID );
      rafID = null;
    }

      function toggleMono() {
        if (audioInput != realAudioInput) {
          audioInput.disconnect();
          realAudioInput.disconnect();
          audioInput = realAudioInput;
        } else {
          realAudioInput.disconnect();
          audioInput = convertToMono( realAudioInput );
        }

        audioInput.connect(inputPoint);
      }

      function gotStream(stream) {
        inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;
    audioInput.connect(inputPoint);

//    audioInput = convertToMono( input );

analyserNode = audioContext.createAnalyser();
analyserNode.fftSize = 2048;
inputPoint.connect( analyserNode );

audioRecorder = new Recorder( inputPoint );

zeroGain = audioContext.createGain();
zeroGain.gain.value = 0.0;
inputPoint.connect( zeroGain );
zeroGain.connect( audioContext.destination );
//updateAnalysers();
handleMicStream(stream);
//setInterval(function(){ wavesurfer.load(stream)}, 3000);
  }

  function loadMedia(selectedMedia, callback) {
    songButton.disabled = false;

    var request = new XMLHttpRequest();
    request.open('GET', selectedMedia.file, true);
    request.responseType = 'arraybuffer';

    request.onload = function() {
      audioContext.decodeAudioData(request.response, function(buffer) {
        var slice = selectedMedia.slice;
        AudioBufferSlice(buffer, slice.start, slice.end, function(error, buf) {
          callback(buf);
        });
      });
    };

    request.send();
  }

  function selectMedia() {
    songButton.disabled = false;
    selectedMedia = media[songSelect.value];
  }

  function playSong() {
    loadMedia(selectedMedia, function(songBuffer) {
      spectro.connectSource(songBuffer, audioContext);
      spectro.start();
    });

    removeControls();
  }

  function requestMic() {
    navigator.getUserMedia({
      video: false,
      audio: true
    },
    function(stream) {
      handleMicStream(stream);
      removeControls();
    }, handleMicError);
  }

  function handleMicStream(stream) {
    var input = audioContext.createMediaStreamSource(stream);
    var analyser = audioContext.createAnalyser();

    analyser.smoothingTimeConstant = 0;
    analyser.fftSize = 2048;

    input.connect(analyser);

    spectro.connectSource(analyser, audioContext);
    spectro.start();
  }

  function handleMicError(error) {
    alert(error);
    console.log(error);
  }

  function removeControls() {
    songSelect.parentNode.removeChild(songSelect);
    songButton.parentNode.removeChild(songButton);
    microphoneButton.parentNode.removeChild(microphoneButton);
  }

  var settings
  var settings2

  function initAudio() {
    if (!navigator.getUserMedia)
      navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!navigator.cancelAnimationFrame)
      navigator.cancelAnimationFrame = navigator.webkitCancelAnimationFrame || navigator.mozCancelAnimationFrame;
    if (!navigator.requestAnimationFrame)
      navigator.requestAnimationFrame = navigator.webkitRequestAnimationFrame || navigator.mozRequestAnimationFrame;

    navigator.getUserMedia(
    {
      "audio": {
        "mandatory": {
          "googEchoCancellation": "false",
          "googAutoGainControl": "false",
          "googNoiseSuppression": "false",
          "googHighpassFilter": "false"
        },
        "optional": []
      },
    }, gotStream, function(e) {
      alert('Error getting audio');
      console.log(e);
    });
    
    init();
}

function init() {
  spectro = Spectrogram(document.getElementById('spectrogram'), {
    canvas: {
      width: function() {
        return window.innerWidth;
      },
      height: 300
    },
    audio: {
      enable: true
    },
    colors: function(steps) {
      var baseColors = [[32,32,32,0], [0,255,255,1], [0,255,0,1], [255,255,0,1], [ 255,0,0,1]];
      //var baseColors = [[32,32,32,0], [32,32,32,0], [32,32,32,0], [32,32,32,0], [ 255,0,0,1]];
      //var baseColors = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [ 255,0,0,1]];
      var positions = [0, 0.15, 0.30, 0.50, 0.75];

      var scale = new chroma.scale(baseColors, positions)
      .domain([0, steps]);

      var colors = [];

      for (var i = 0; i < steps; ++i) {
        var color = scale(i);
        colors.push(color.hex());
      }

      return colors;
    }
  });

  try {
    audioContext = new AudioContext();
  } catch (e) {
    alert('No web audio support in this browser!');
  }
  
}

jQuery(document).ready(function($) {
  var $tabButtonItem = $('#tab-button li'),
      $tabSelect = $('#tab-select'),
      $tabContents = $('.tab-contents'),
      activeClass = 'is-active';

  $tabButtonItem.first().addClass(activeClass);
  $tabContents.not(':first').hide();

  $tabButtonItem.find('a').on('click', function(e) {
    var target = $(this).attr('href');

    $tabButtonItem.removeClass(activeClass);
    $(this).parent().addClass(activeClass);
    $tabSelect.val(target);
    $tabContents.hide();
    $(target).show();
    e.preventDefault();
  });

  $tabSelect.on('change', function() {
    var target = $(this).val(),
        targetSelectNum = $(this).prop('selectedIndex');

    $tabButtonItem.removeClass(activeClass);
    $tabButtonItem.eq(targetSelectNum).addClass(activeClass);
    $tabContents.hide();
    $(target).show();
  });

});


window.addEventListener('load', initAudio );
