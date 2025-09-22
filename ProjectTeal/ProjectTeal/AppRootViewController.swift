//
//  AppRootViewController.swift
//  ProjectTeal
//
//  Created by Soleil Yu on 2025/9/23.
//

import UIKit

class AppRootViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set a default background color
        view.backgroundColor = .systemBackground

        // Embed camera view controller full-screen
        let cameraVC = CameraViewController()
        addChild(cameraVC)
        cameraVC.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(cameraVC.view)
        NSLayoutConstraint.activate([
            cameraVC.view.topAnchor.constraint(equalTo: view.topAnchor),
            cameraVC.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            cameraVC.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            cameraVC.view.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
        cameraVC.didMove(toParent: self)
    }
}
