function addToList(){
    var node = document.createElement('li');
  node.appendChild(document.createTextNode("Name: " + document.getElementById("appleForm").name.value + " ............. Favorite Apple: " + document.getElementById("appleForm").fapple.value));
   
  document.getElementById("appleList").appendChild(node);
  }