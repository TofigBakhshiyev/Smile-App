import React, { Component } from "react";
import Emoticons from "./Emoticons"; 
import mqtt from "mqtt";

import "./App.css"; 

class App extends Component {
  constructor() {
    super();
    this.state = { 
      emotion_numbers: 4,
      data_data: []
    };
  } 
  componentDidMount() {
    // connecting Mosca server
    let client = mqtt.connect("ws://localhost:3002")
    client.on("connect", () => {
      // subscribe to every possible topic
      client.subscribe("person"); 
        console.log("connected ");
      }); 

      // definning variables for message
      let emotions
      let emotion_numbers = new Map()
      let ratings = ['anger', 'sad', 'neutral', 'surprise', 'happy']

      // assign number for every emotion
      for (let i = 0; i < ratings.length; i++) {
        emotion_numbers[ratings[i]] = i + 1
      }   
      
      client.on("message", function (topic, message) {
        emotions = JSON.parse(message)  
        // Updates React state with message  
        this.handleMqttData(emotion_numbers[emotions.emotion])  
        this.setDatas(emotions)
      }.bind(this))  
  }

  handleMqttData = (i) => {
    this.setState({emotion_numbers: i})
  }  

  setDatas = (data) => {
    this.setState({ data_data: data})
  }
 
  render() {
    const { emotion_numbers } = this.state;
    
    // changing colors for every emotion
    let background = "";
    switch (emotion_numbers) {
      case 1:
        background = "shock-face";
        break;
      case 2:
        background = "sad-face";
        break;
      case 3:
        background = "flat-face";
        break;
      case 4:
        background = "smile-face";
        break;
      case 5:
        background = "happy-face";
        break;
      default:
        background = "flat-face";
    }

    return (
      <div className="appC">
        <div className={`App ${background}`}>
          <Emoticons selected_rating={emotion_numbers} />  
          <p>
            Age { this.state.data_data.age } <br/> 
            Gender { this.state.data_data.gender } <br/> 
            Emotion { this.state.data_data.emotion } 
          </p> 
        </div>  
      </div> 
    );
  }
}

export default App;
