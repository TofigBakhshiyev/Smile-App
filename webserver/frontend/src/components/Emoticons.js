import React from "react";


const Emoticons = (props) => {
  const { selected_rating } = props;

  let mouth = "";
  switch (selected_rating) {
    case 1:
      mouth = "shock";
      break;
    case 2:
      mouth = "sad";
      break;
    case 3:
      mouth = "flat";
      break;
    case 4:
      mouth = "smile";
      break;
    case 5:
      mouth = "happy";
      break;
    default:
      mouth = "smile";
  }

  return (
    <div className="face">
      <div className="eye" />
      <div className="mouth">
        <div className={mouth} />
      </div>
      <div className="eye" />
    </div>
  );
};

export default Emoticons;
