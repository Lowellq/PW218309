window.onload=function(){

	var operandor="";
	var  numeros=function(){
		var valor=this.value;

		if(operandor==""){//Operando 1
			var valorInput=document.getElementById("operando1").value;

			if (valorInput=="0") {
				document.getElementById("operando1").value="";
			}
			document.getElementById("operando1").valuet=valor;
		}
		else { //Operando 2
			var valorInput=document.getElementById("operando2").value;

			if (valorInput=="0") {
				document.getElementById("operando2").value="";
			}
			document.getElementById("operando2").valuet=valor;
		}
	}




var ColorAmarillo=function(){
	this.style.background="yellow";

}

var ColorBlanco=function(){
	this.style.background="white";

}
	
var operando1=document.getElementById("operando1")
operando1.addEventListener("focus", ColorAmarillo)
operando1.addEventListener("focusout", ColorBlanco)
}