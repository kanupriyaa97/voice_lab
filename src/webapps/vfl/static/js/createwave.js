var container = $('#wave');
var siriWave;

var addWave = function() {
	siriWave = new SiriWave({
		container: container[0],
		width: container.width(),
		height: 80,
		speed: 0.05,
		amplitude: 1,
		autostart: true,
	});
}
addWave();

$(window).resize(function() {
	siriWave.stop();
	$('#wave canvas').remove();
	addWave();
});

setInterval(function(){ siriWave.setSpeed(0.05); }, 5000);