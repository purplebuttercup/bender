var express = require('express'),
    http = require('http'),
    path = require('path'),
    bodyParser= require('body-parser'),
    spawn = require('child_process').spawn,
    sys = require('sys');

var app = express();

app.set('port', 3000);

app.use(express.static(path.normalize(__dirname + '/')));
app.use(bodyParser.json()); // for parsing application/json
app.use(bodyParser.urlencoded({ extended: true })); // for parsing       application/x-www-form-urlencoded



http.createServer(app).listen(app.get('port'), function() {
    console.log('Express server listening on port ' + app.get('port'));
});


app.post('/view1', function(req, res) {
    console.log(req.body.sentence);

    var python = spawn(
        'python3',
        ['Evaluation.py', req.body.sentence],
        {
            cwd: '../../PycharmProjects/text-classification-tf-eval/'
        }
    )

    python.on('error', function( err ){ console.log(err); });

    python.stdout.on('data', function (data) {
        console.log('stdout: ' + data);
        res.end(data);
    });

    python.stderr.on('data', function (data) {
        console.log('stderr: ' + data);
    });

    python.on('close', function (code) {
        console.log('child process exited with code ' + code);
    });

    // ex

});