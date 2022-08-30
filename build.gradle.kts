plugins {
    id("java")
    application
}

group = "me.neuralnetwork"
version = "0.1"

dependencies {
    implementation(fileTree("libs") { include("*.jar") })
}

application {
    mainClass.set("me.neuralnetwork.nn.App")
}

