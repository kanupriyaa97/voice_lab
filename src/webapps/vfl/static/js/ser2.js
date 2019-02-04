var initial_link = document.getElementById('download');
initial_link.style.visibility = "hidden";

URL = window.URL || window.webkitURL;

var gumStream; //getUserMedia() stream
var rec; //Recorder.js object
var input; //MediaStreamAudioSourceNode

var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext = new AudioContext; //new audio context to help us record

var recordButton = document.getElementById("kanupriyaa");
var stopButton = document.getElementById("eshwarya");
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);


function startRecording () {
    var constraints = { audio: true, video:false }

    recordButton.disabled = true;
    stopButton.disabled = false;

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        gumStream = stream;
        input = audioContext.createMediaStreamSource(stream);
        rec = new Recorder(input,{numChannels:1});
        rec.record();

    }).catch(function(err) {
        recordButton.disabled = false;
        stopButton.disabled = true
    });

}

function stopRecording() {
    stopButton.disabled = true;
    recordButton.disabled = false;

    rec.stop();
    gumStream.getAudioTracks()[0].stop();
    rec.exportWAV(makeResults);

}

//Taken from a stackoverflow question
function getCookie(c_name)
{
    if (document.cookie.length > 0)
    {
        c_start = document.cookie.indexOf(c_name + "=");
        if (c_start != -1)
        {
            c_start = c_start + c_name.length + 1;
            c_end = document.cookie.indexOf(";", c_start);
            if (c_end == -1) c_end = document.cookie.length;
            return unescape(document.cookie.substring(c_start,c_end));
        }
    }
    return "";
 }

function makeResults(blob) {
    var url = URL.createObjectURL(blob);
    var link = document.getElementById('download');
    link.style.visibility = "visible";
    link.href = url;
    link.download = new Date().toISOString() + '.wav';

    var wavesurfer = WaveSurfer.create({
        container: '#waveform',
        scrollParent: false,
        barHeight: 2,
        barWidth:2,
        height: 300,
        hideScrollbar: true,
        waveColor: 'red',
        progressColor: 'blue'
    })

    wavesurfer.load(url);
    wavesurfer.on('ready', function () {
        wavesurfer.play();
    });


    var xhttp = new XMLHttpRequest();
    xhttp.open("POST", "/vfl/texttospeech", true);
    xhttp.setRequestHeader('X-CSRFToken', getCookie("csrftoken"));
    var data = new FormData();
    data.append("audio", blob, 'audio.wav');
    xhttp.send(data);


    xhttp.onload = function() {
        var result = JSON.parse(xhttp.response);

        for (var key in result) {
            var tr = "<tr>";
            tr += "<td>" + key + "</td>";
            tr += "<td>" + result[key].toString() + "</td>"
            tr += "</tr>";
            document.getElementById("chars_tbody").innerHTML += tr;
        }
    };
   }

