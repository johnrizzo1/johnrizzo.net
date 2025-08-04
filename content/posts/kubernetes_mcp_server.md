+++
draft = false
authors = ["John Rizzo"]
title = "Enabling AI with Kubernetes MCP Server"
# slug = "virtualization-lab"
date = "2025-08-03"
tags = [
  "computer science",
  "artificial intelligence"
]
categories = [
  "computer science",
  "artificial intelligence"
]
series = [
  "MCP Servers"
]
+++

Kubernetes is a powerful platform for deploying and managing applications in a containerized environment. It provides the tools to automate deployment, scaling, and operations of application containers across clusters of hosts. This post will be short as I wanted to show off my latest creation and get your feedback.  I have created a Kubernetes MCP server that allows tools such as Claude Code to interact with Kubernetes clusters. This MCP server is designed to be a bridge between AI tools and Kubernetes, enabling seamless integration and management of containerized applications.

The MCP server is built using Python and Flask, providing a RESTful API that allows AI tools to interact with Kubernetes clusters. Modeled after helm and kubectl, it supports various operations such as deploying applications, scaling services, and monitoring cluster health.

[Here is the GitHub repository for the MCP server](https://github.com/johnrizzo1/ubermcp)

{{< youtube duAyM3XTFww >}}
