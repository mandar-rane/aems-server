const http = require('http');
const express = require('express');
const { Server } = require('socket.io');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
app.use(cors());
const server = http.createServer(app);
const io = new Server(server);

const scriptName = "HeadPoseScript.py";
const pythonInterpreter = path.join(__dirname, 'headpenv', 'Scripts', 'python');

let headPoseProcess;

io.on('connection', (socket) => {
    console.log('Client Connected');

    socket.on('update_variable', (data) => {
        console.log(`Received updated variable from Python client :${data}`);
        io.emit('update_variable', data);
    })

    socket.on('attendance', (data) => {
        console.log(`Received attendance from Python client :${data}`);
        io.emit('attendance', data);
    })


    socket.on('disconnect', () => {
        console.log('User disconnected');
    }); 
});

const port = process.env.PORT || 3000;

app.post('/start-python-script', (req, res) => {
    headPoseProcess = spawn(pythonInterpreter, [scriptName]);

    headPoseProcess.stdout.on('data', (data) => {
        console.log(`Python script output: ${data}`);
    });

    headPoseProcess.stderr.on('data', (data) => {
        console.error(`Error executing Python script: ${data}`);
    });

    headPoseProcess.on('close', (code) => {
        console.log(`Python script process exited with code ${code}`);
        res.send('Python script execution completed.');
    });

})


app.post('/stop-script', (req, res) => {
    
    if (headPoseProcess) {
        // Terminate the Python script process
        headPoseProcess.kill('SIGINT'); // Send interrupt signal
        res.send('Script execution stopped.');
    } else {
        res.status(400).send('No script running to stop.');
    }
});


server.listen(port, () => {
    console.log(`Server Listening on Port: ${port}`);
});








