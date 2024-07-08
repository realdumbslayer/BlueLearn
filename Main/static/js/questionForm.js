const types = [
    "Short Answer",
    "Multiple Choice"
  ];
  
  const allFields = document.querySelector("#allFields");
  let questions = JSON.parse(localStorage.getItem("Questions")) || [];
  
  let queCount = localStorage.getItem("queCount") || 0;
  
  // When the input of the question is changed
  const changeOption = (id) => {
    const select = "select" + id;
    const selectData = document.getElementById(select).value;
    const index = questions.findIndex((que) => que.id == id);
    questions[index].type = selectData;
    if (
      (selectData == 1) &&
      questions[index].options.length == 0
    ) {
      questions[index].options = ["Option 1"];
    }
    // Update data in the local storage
    updateLocal();
    // Refresh the list of questions
    drawAllFields();
  };
  
  const changeTitle = (id) => {
    const title = "title" + id;
    const index = questions.findIndex((que) => que.id == id);
    questions[index].title = document.getElementById(title).value;
    updateLocal();
  };
  
  const changeInputOption = (id, index) => {
    const option = "option" + id + "_" + index;
    const queIndex = questions.findIndex((que) => que.id == id);
    questions[queIndex].options[index] = document.getElementById(option).value;
    updateLocal();
  };
  
  const addOption = (id) => {
    const index = questions.findIndex((que) => que.id == id);
    questions[index].options.push(
      "Option " +
        (parseInt(questions[index].options.length) + parseInt(1))
    );
    updateLocal();
    drawAllFields();
  };
  
  const removeQue = (id) => {
    const index = questions.findIndex((que) => que.id == id);
    questions.splice(index, 1);
    // Set Question Count to 0 when no question left
    if (questions.length === 0) {
      localStorage.setItem("queCount", 0);
      queCount = 0;
    }
    drawAllFields();
  };
  
  const removeOption = (index, id) => {
    const queIndex = questions.findIndex((que) => que.id == id);
    questions[queIndex].options.splice(index, 1);
    updateLocal();
    drawAllFields();
  };
  
  const updateLocal = () => {
    localStorage.setItem("Questions", JSON.stringify(questions));
  };
  
  const addField = () => {
    let data = {
      id: queCount,
      title: "",
      type: 0,
      options: [],
      required: false,
    };
    questions.push(data);
    queCount++;
    localStorage.setItem("queCount", queCount);
    drawAllFields();
  };
  
  const drawAllFields = () => {
    allFields.innerHTML = "";
    localStorage.setItem("Questions", JSON.stringify(questions));
    questions = JSON.parse(localStorage.getItem("Questions"));
  
    questions.forEach((que) => {
      let options = "<option hidden>Select Type</option>";
      for (let i = 0; i < 2; i++) {
        if (i == que.type) {
          options += `<option selected value="${i}">${types[i]}</option>`;
        } else {
          options += `<option value="${i}">${types[i]}</option>`;
        }
      }
      let output = "";
      let addOptionBtn = "";
      if (que.type == 0) {
        output = `<input class='textInput' type='text' placeholder='Short answer text' disabled/>`;
      } else if (que.type == 1) {
        // Multiple Choice Question
        for (let i = 0; i < que.options.length; i++) {
          let a = i + 1;
          output += `<div class='optionInput'>
                      <input class='radio' type='radio' disabled/> 
                      <input oninput="changeInputOption(${que.id},${i})" type='text' class='addInput'placeholder='Option ${a}' id='option${que.id}_${i}' value='${que.options[i]}'/> 
                      <span class="delete material-symbols-outlined" onclick='removeOption(${i},${que.id})'>close</span>
                  </div>`;
        }
        addOptionBtn = `<button class='addOption' onclick="addOption(${que.id})">Add option</button>`;
      }
  
      allFields.innerHTML += `
              <div class="card">
                  <div class="inputData">
                      <input id='title${que.id}' oninput="changeTitle(${que.id})" type="text" placeholder="Question"  value='${que.title}' />
                      <select id="select${que.id}" onchange="changeOption(${que.id})" >
                          ${options}
                      </select>
                  </div>
                  <div class="output">
                      ${output}
                      ${addOptionBtn}
                  </div>
                  <div class='cardFooter'>
                      <span class="close material-symbols-outlined" onclick='removeQue(${que.id})'>
                      delete</span>
                  </card>
                  
          </div>`;
    });
  };
  
  drawAllFields();
  