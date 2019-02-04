const recordAudio = () => {
  return new Promise(resolve => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const mediaRecorder = new MediaRecorder(stream);
        const audioChunks = [];

        mediaRecorder.addEventListener("dataavailable", event => {
          audioChunks.push(event.data);
        });

        const start = () => {
          mediaRecorder.start();
        };

        const stop = () => {
          return new Promise(resolve => {
            mediaRecorder.addEventListener("stop", () => {
              const audioBlob = new Blob(audioChunks);
              resolve({ audioBlob});

              var arrayBuffer = new ArrayBuffer();
              var fileReader = new FileReader();
              fileReader.onload = function(event) {
                    arrayBuffer = event.target.result;
              };
              fileReader.readAsArrayBuffer(audioBlob);

              make_spectrogram(arrayBuffer);
              var xhr = new XMLHttpRequest();
              xhr.open('POST', 'recordaudio', true);
              var csrftoken = Cookies.get('csrftoken');
              xhr.setRequestHeader('X-CSRFToken', csrftoken);
              var fd = new FormData();
              fd.append("audio_file", audioBlob);
              xhr.send(fd);
            });
            mediaRecorder.stop();
          });
        };

        resolve({ start, stop });
      });
  });
};

async function record(){
    console.log("recording");
  const recorder = await recordAudio();
  console.log("awaited record audio");
  recorder.start();
  console.log("started");

  setTimeout(async () => {
    const audio = await recorder.stop();
  }, 3000);
  console.log("completed");
};










function make_spectrogram (arrayBuffer) {
   console.log("inside spectrogram");
    // check if the default naming is enabled, if not use the chrome one.
    if (! window.AudioContext) {
        if (! window.webkitAudioContext) {
            alert('no audiocontext found');
        }
        window.AudioContext = window.webkitAudioContext;
    }
    var context = new AudioContext();
    var audioBuffer;
    var sourceNode;

    // load the sound
    function loadSound() {
        var request = new XMLHttpRequest();
        request.open('GET','retrieveaudio/1', true);
        request.responseType = 'arraybuffer';
        request.onload = function () {
            var audioData = request.response;
            context.decodeAudioData(audioData, function(buffer) {
                // when the audio is decoded
                audioBuffer = buffer;
            }, onError);
        }
        request.send();
    }
    loadSound();
    setupAudioNodes();


    function onError(e) {
        console.log(e);
    }

    // when the javascript node is called
    // we use information from the analyzer node
    // to draw the volume
    javascriptNode.onaudioprocess = function () {
        console.log("jsNode");
        // get the average for the first channel
        var array = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(array);

        // draw the spectrogram
        if (sourceNode.playbackState == sourceNode.PLAYING_STATE) {
            drawSpectrogram(array);
        }
    }

    function drawSpectrogram(array) {
        console.log("draw spec");
         // create a temp canvas we use for copying and scrolling
        var tempCanvas = document.createElement("canvas");
        tempCtx = tempCanvas.getContext("2d");
        tempCanvas.width=800;
        tempCanvas.height=512;


        // used for color distribution
        var hot = new chroma.scale({
            colors:['#000000', '#ff0000', '#ffff00', '#ffffff'],
            positions:[0, .25, .75, 1],
            mode:'rgb',
            limits:[0, 300]
        });

        // copy the current canvas onto the temp canvas
            tempCtx.drawImage(tempCanvas, 0, 0, 800, 512);

            // iterate over the elements from the array
            for (var i = 0; i < 4; i++) {
                // draw each pixel with the specific color
                var value = array[i];
                tempCtx.fillStyle = hot.colors(value).hex();

                // draw the line at the right side of the canvas
                tempCtx.fillRect(800 - 1, 512 - i, 1, 1);


            // set translate on the canvas
            tempCtx.translate(-1, 0);
            // draw the copied image
            tempCtx.drawImage(tempCanvas, 0, 0, 800, 512, 0, 0, 800, 512);

            // reset the transformation matrix
            tempCtx.setTransform(1, 0, 0, 1, 0, 0);
            console.log("ends");
        };
    }

    function setupAudioNodes() {
        console.log("setnodes");
        // setup a javascript node
        javascriptNode = context.createScriptProcessor(2048, 1, 1);
        // connect to destination, else it isn't called
        javascriptNode.connect(context.destination);

        // setup a analyzer
        analyser = context.createAnalyser();
        analyser.smoothingTimeConstant = 0.3;
        analyser.fftSize = 1024;

        // create a buffer source node
        sourceNode = context.createBufferSource();
        sourceNode.buffer = audioBuffer;

        // connect the source to the analyser
        sourceNode.connect(analyser);

        // we use the javascript node to draw at a specific interval.
        analyser.connect(javascriptNode);

        // and connect to destination, if you want audio
        //sourceNode.connect(context.destination);
    }

};