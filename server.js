const http = require('http');
const express = require('express');
const { Server } = require('socket.io');
const cors = require('cors');

const app = express();
app.use(cors());
const server = http.createServer(app);
const io = new Server(server);

io.on('connection', (socket) => {
    console.log('Client Connected');

    socket.on('update_variable', (data) => {
        console.log(`Received updated variable from Python client :${data}`);
        io.emit('update_variable', data);
    })
    socket.on('disconnect', () => {
        console.log('User disconnected');
    }); 
});

const port = process.env.PORT || 3000;


server.listen(port, () => {
    console.log(`Server Listening on Port: ${port}`);
});








