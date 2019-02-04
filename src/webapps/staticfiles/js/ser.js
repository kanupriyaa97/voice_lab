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

              var xhr = new XMLHttpRequest();
              xhr.open('POST', 'ser', true);
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
  const recorder = await recordAudio();
  recorder.start();

  setTimeout(async () => {
    const audio = await recorder.stop();
  }, 5000);
  console.log("completed");

     var wavesurfer = WaveSurfer.create({
        container: '#waveform',
        scrollParent: false,
        barHeight: 2,
        barWidth:2,
        height: 300,
        hideScrollbar: true,
        waveColor: 'red',
        progressColor: 'green'
    })

    wavesurfer.load('../vfl/waveformaudio/8');
    wavesurfer.on('ready', function () {
        wavesurfer.play();
    });
    console.log("wave");
};