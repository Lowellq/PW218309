//Cargar la aplicacion de electron
const app=require ('electron').app;

//Crear ventanas del sistema operativo
const BrowserWindow=require('electron').BrowserWindow;

//Ruta del sistema de archivo del S.O
const path=require('path');
const url=require('url');

//Otra forma de declarar una constante
//Pantalla principal
let PantallaPrincial;


function MustraPnatallaPrincipal(){
	//Creamos una pantalla vacia
	PantallaPrincial=new BrowserWindow ({width:300,height:420});

	//Cargamos en la pantalla el contenido de nuestra pagina
	PantallaPrincial.loadURL(url.format({
		pathname: path.join(__dirname, 'index.html'),
		protocol: 'file',
		slashes: true

	}));
	//Mostraremos la pantalla
	PantallaPrincial.show();
}

app.on('ready'),MuestraPantallaPrincipal(); 